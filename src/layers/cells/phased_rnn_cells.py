"""
This file implements various Phased RNN cells. These
cells are directly based on the Phased LSTM design described
in the paper below.
https://papers.nips.cc/paper/6310-phased-lstm-accelerating-recurrent-network-training-for-long-or-event-based-sequences.pdf

We also use the following repository as a model for this implementation.
https://github.com/philipperemy/tensorflow-phased-lstm
"""
import tensorflow as tf
from collections import namedtuple
from typing import Optional, Any, Tuple

from utils.tfutils import get_activation
from utils.constants import SMALL_NUMBER, ONE_HALF


PhasedUGRNNStateTuple = namedtuple('PhasedUGRNNStateTuple', ['state', 'time'])
PhasedUGRNNOutputTuple = namedtuple('PhasedUGRNNOutputTuple', ['output', 'time_gate'])


def phi(time: tf.Tensor, shift: tf.Tensor, period: tf.Tensor) -> tf.Tensor:
    return tf.div(tf.mod(time - shift, period), period)


def time_gate(time: tf.Tensor, shift: tf.Tensor, on_fraction: tf.Tensor, period: tf.Tensor, leak_rate: tf.Tensor) -> tf.Tensor:
    phi_t = phi(time=time, shift=shift, period=period)

    half_on_fraction = ONE_HALF * on_fraction

    mask_1 = tf.cast(phi_t < half_on_fraction, dtype=tf.float32)
    mask_2 = tf.cast(tf.logical_and(phi_t >= half_on_fraction, phi_t < on_fraction), dtype=tf.float32)
    mask_3 = tf.cast(phi_t >= on_fraction, dtype=tf.float32)

    linear_1 = 2 * phi_t / on_fraction
    linear_2 = 2 - linear_1
    linear_3 = leak_rate * phi_t

    return mask_1 * linear_1 + mask_2 * linear_2 + mask_3 * linear_3


class PhasedUGRNNCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self,
                 units: int,
                 activation: str,
                 on_fraction: tf.Tensor,
                 period_init: tf.Tensor,
                 leak_rate: tf.Tensor,
                 name: str):
        self._units = units
        self._activation = get_activation(activation)
        self._on_fraction = on_fraction
        self._leak_rate = leak_rate

        # Make the trainable variables for this cell
        self.W_update = tf.get_variable(name='{0}-W-update-kernel'.format(name),
                                        initializer=tf.glorot_uniform_initializer(),
                                        shape=[2 * units, units],
                                        trainable=True)
        self.b_update = tf.get_variable(name='{0}-b-update-bias'.format(name),
                                        initializer=tf.glorot_uniform_initializer(),
                                        shape=[1, units],
                                        trainable=True)

        self.W = tf.get_variable(name='{0}-W-kernel'.format(name),
                                 initializer=tf.glorot_uniform_initializer(),
                                 shape=[2 * units, units],
                                 trainable=True)
        self.b = tf.get_variable(name='{0}-b-bias'.format(name),
                                 initializer=tf.glorot_uniform_initializer(),
                                 shape=[1, units],
                                 trainable=True)

        self.period = tf.get_variable(name='{0}-period'.format(name),
                                      initializer=tf.random_uniform_initializer(minval=0.0, maxval=period_init, dtype=tf.float32),
                                      shape=[],
                                      trainable=True)

        self.shift = tf.get_variable(name='{0}-shift'.format(name),
                                     initializer=tf.random_uniform_initializer(minval=0.0, maxval=self.period.initialized_value(), dtype=tf.float32),
                                     shape=[],
                                     trainable=True)

    @property
    def state_size(self) -> PhasedUGRNNStateTuple:
        return PhasedUGRNNStateTuple(self._units, 1)

    @property
    def output_size(self) -> PhasedUGRNNOutputTuple:
        return PhasedUGRNNOutputTuple(self._units, 1)

    def get_initial_state(self, inputs: Optional[tf.Tensor], batch_size: Optional[int], dtype: Any) -> PhasedUGRNNStateTuple:
        """
        Creates an initial state by setting the hidden state to zero and the update probability to 1.
        """
        initial_state = tf.get_variable(name='initial-hidden-state',
                                        initializer=tf.zeros_initializer(),
                                        shape=[1, self._units],
                                        dtype=dtype,
                                        trainable=False)
        initial_time = tf.get_variable(name='initial-time',
                                       initializer=tf.zeros_initializer(),
                                       shape=[1, 1],
                                       dtype=dtype,
                                       trainable=False)

        # We tile the initial states across the entire batch
        return PhasedUGRNNStateTuple(state=tf.tile(initial_state, multiples=(batch_size, 1)),
                                     time=tf.tile(initial_time, multiples=(batch_size, 1)))

    def __call__(self, inputs: tf.Tensor, state: PhasedUGRNNStateTuple, scope: Optional[str] = None) -> Tuple[PhasedUGRNNOutputTuple, PhasedUGRNNStateTuple]:
        # Unpack the previous state
        prev_state, time = state

        scope = scope if scope is not None else type(self).__name__
        with tf.variable_scope(scope):
            # Concatenate the previous state and the current input. This allows for fewer matrix multiplications
            # and helps with efficiency
            input_state_concat = tf.concat([prev_state, inputs], axis=-1)  # [B, 2 * D]

            # Compute the standard UGRNN Cell
            update_state = tf.matmul(input_state_concat, self.W_update)  # [B, D]
            update = tf.math.sigmoid(update_state + self.b_update + 1)

            candidate_state = tf.matmul(input_state_concat, self.W)  # [B, D]
            candidate = self._activation(candidate_state + self.b)

            next_cell_state = update * candidate + (1 - update) * prev_state  # [B, D]

            # Apply the time oscillation gate
            kt = time_gate(time=time, period=self.period, on_fraction=self._on_fraction, shift=self.shift, leak_rate=self._leak_rate)
            next_state = kt * next_cell_state + (1 - kt) * prev_state

            phased_state = PhasedUGRNNStateTuple(next_state, time + 1)
            phased_output = PhasedUGRNNOutputTuple(next_state, kt)

        return phased_output, phased_state
