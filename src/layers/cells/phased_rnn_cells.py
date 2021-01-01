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

from utils.tfutils import get_activation, apply_noise
from utils.constants import SMALL_NUMBER, ONE_HALF
from .cell_utils import ugrnn


PhasedUGRNNStateTuple = namedtuple('PhasedUGRNNStateTuple', ['state', 'time'])
PhasedUGRNNOutputTuple = namedtuple('PhasedUGRNNOutputTuple', ['output', 'time_gate'])


def phi(time: tf.Tensor, shift: tf.Tensor, period: tf.Tensor) -> tf.Tensor:
    return tf.math.divide(tf.math.mod(time - shift, period), period)


def time_gate(time: tf.Tensor, shift: tf.Tensor, on_fraction: tf.Tensor, period: tf.Tensor, leak_rate: tf.Tensor) -> tf.Tensor:
    phi_t = phi(time=time, shift=shift, period=period)

    half_on_fraction = ONE_HALF * on_fraction

    mask_1 = tf.cast(tf.less_equal(phi_t, half_on_fraction), dtype=tf.float32)
    mask_2 = tf.cast(tf.logical_and(tf.less(half_on_fraction, phi_t), tf.less(phi_t, on_fraction)), dtype=tf.float32)
    mask_3 = tf.cast(tf.greater_equal(phi_t, on_fraction), dtype=tf.float32)

    term_1 = tf.multiply(mask_1, 2 * phi_t / on_fraction)
    term_2 = tf.multiply(mask_2, 2 - 2 * phi_t / on_fraction)
    term_3 = tf.multiply(mask_3, leak_rate * phi_t)

    return term_1 + term_2 + term_3


class PhasedUGRNNCell(tf.compat.v1.nn.rnn_cell.RNNCell):

    def __init__(self,
                 units: int,
                 activation: str,
                 on_fraction: tf.Tensor,
                 period_init: tf.Tensor,
                 leak_rate: tf.Tensor,
                 recurrent_noise: tf.Tensor,
                 name: str):
        self._units = units
        self._activation = get_activation(activation)
        self._on_fraction = on_fraction
        self._leak_rate = leak_rate
        self._recurrent_noise = recurrent_noise

        # Make the trainable variables for this cell
        self.W_transform = tf.compat.v1.get_variable(name='{0}-W-transform'.format(name),
                                                     initializer=tf.compat.v1.glorot_uniform_initializer(),
                                                     shape=[2 * units, 2 * units],
                                                     trainable=True)
        self.b_transform = tf.compat.v1.get_variable(name='{0}-b-transform'.format(name),
                                                     initializer=tf.compat.v1.glorot_uniform_initializer(),
                                                     shape=[1, 2 * units],
                                                     trainable=True)

        # The original Phased LSTM uses a [D] dimensional time gate. This gates each dimension
        # of the hidden state using a different period and shift (though it potentially uses the
        # same on fraction). Using different periods for each dimension saves computation, but
        # the lack of alignment means that we still may process all inputs. We target applications
        # in which capturing inputs is expensive, so we need to align the time gate across all dimensions
        # to ensure that entire inputs are skipped.
        self.period = tf.compat.v1.get_variable(name='{0}-period'.format(name),
                                                initializer=tf.compat.v1.random_uniform_initializer(minval=0.0, maxval=period_init, dtype=tf.float32),
                                                shape=[],
                                                trainable=True)

        self.shift = tf.compat.v1.get_variable(name='{0}-shift'.format(name),
                                               initializer=tf.compat.v1.random_uniform_initializer(minval=0.0, maxval=self.period.initialized_value(), dtype=tf.float32),
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
        initial_state = tf.compat.v1.get_variable(name='initial-hidden-state',
                                                  initializer=tf.compat.v1.zeros_initializer(),
                                                  shape=[1, self._units],
                                                  dtype=dtype,
                                                  trainable=False)
        initial_time = tf.compat.v1.get_variable(name='initial-time',
                                                 initializer=tf.compat.v1.zeros_initializer(),
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
        with tf.compat.v1.variable_scope(scope):

            # Apply the standard UGRNN update, [B, D]
            next_cell_state = ugrnn(inputs=inputs,
                                    state=prev_state,
                                    W_transform=self.W_transform,
                                    b_transform=self.b_transform,
                                    activation=self._activation)

            # Apply regularization noise
            next_cell_state = apply_noise(next_cell_state, scale=self._recurrent_noise)

            # Apply the time oscillation gate
            kt = time_gate(time=time, period=self.period, on_fraction=self._on_fraction, shift=self.shift, leak_rate=self._leak_rate)
            next_state = kt * next_cell_state + (1 - kt) * prev_state

            phased_state = PhasedUGRNNStateTuple(next_state, time + 1)
            phased_output = PhasedUGRNNOutputTuple(next_state, kt)

        return phased_output, phased_state
