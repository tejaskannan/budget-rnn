"""
This file implements various Skip RNN Cells. The classes here
are inspired by the following repository and associated paper:

Paper: https://arxiv.org/abs/1708.06834
Repo: https://github.com/imatge-upc/skiprnn-2017-telecombcn/blob/master/src/rnn_cells/skip_rnn_cells.py.
"""
import tensorflow as tf
from collections import namedtuple
from typing import Optional, Any, Tuple

from utils.tfutils import get_activation, apply_noise
from utils.constants import SMALL_NUMBER
from .cell_utils import ugrnn


SkipUGRNNStateTuple = namedtuple('SkipUGRNNStateTuple', ['state', 'cumulative_state_update'])
SkipUGRNNOutputTuple = namedtuple('SkipUGRNNOutputTuple', ['output', 'state_update_gate', 'gate_value'])


def binarize(x: tf.Tensor, name: str = 'binarize') -> tf.Tensor:
    """
    Maps the values in the given tensor to {0, 1} using a rounding function. This function
    assigns the gradient to be the identity.
    """
    g = tf.compat.v1.get_default_graph()

    with g.gradient_override_map({'Round': 'Identity'}):
        return tf.round(x, name=name)


class SkipUGRNNCell(tf.compat.v1.nn.rnn_cell.RNNCell):

    def __init__(self, units: int, activation: str, name: str, recurrent_noise: tf.Tensor):
        self._units = units
        self._activation = get_activation(activation)
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
        self.W_state = tf.compat.v1.get_variable(name='{0}-W-state'.format(name),
                                                 initializer=tf.compat.v1.glorot_uniform_initializer(),
                                                 shape=[units, 1],
                                                 trainable=True)
        self.b_state = tf.compat.v1.get_variable(name='{0}-b-state'.format(name),
                                                 initializer=tf.compat.v1.glorot_uniform_initializer(),
                                                 shape=[1, 1],
                                                 trainable=True)

    @property
    def state_size(self) -> SkipUGRNNStateTuple:
        return SkipUGRNNStateTuple(self._units, 1)

    @property
    def output_size(self) -> SkipUGRNNOutputTuple:
        return SkipUGRNNOutputTuple(self._units, 1, 1)

    def get_initial_state(self, inputs: Optional[tf.Tensor], batch_size: Optional[int], dtype: Any) -> SkipUGRNNStateTuple:
        """
        Creates an initial state by setting the hidden state to zero and the update probability to 1.
        """
        initial_state = tf.compat.v1.get_variable(name='initial-hidden-state',
                                                  initializer=tf.compat.v1.zeros_initializer(),
                                                  shape=[1, self._units],
                                                  dtype=dtype,
                                                  trainable=False)
        initial_state_update_prob = tf.compat.v1.get_variable(name='initial-state-update-prob',
                                                              initializer=tf.compat.v1.ones_initializer(),
                                                              shape=[1, 1],
                                                              dtype=dtype,
                                                              trainable=False)

        # We tile the initial states across the entire batch
        return SkipUGRNNStateTuple(state=tf.tile(initial_state, multiples=(batch_size, 1)),
                                   cumulative_state_update=tf.tile(initial_state_update_prob, multiples=(batch_size, 1)))

    def __call__(self, inputs: tf.Tensor, state: SkipUGRNNStateTuple, scope=None) -> Tuple[SkipUGRNNOutputTuple, SkipUGRNNStateTuple]:
        # Unpack the previous state
        prev_state, prev_cum_state_update_prob = state

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

            # Apply the state update gate. This is the Skip portion.
            # We first compute the state update gate. This is a binary version of the cumulative state update prob.
            state_update_gate = binarize(prev_cum_state_update_prob)  # A [B, 1] binary tensor

            # Apply the binary state update gate to get the next state, [B, D]
            next_state = state_update_gate * next_cell_state + (1 - state_update_gate) * prev_state

            # Compute the next state update probability (clipped into the range [0, 1])
            delta_state_update_prob = tf.math.sigmoid(tf.matmul(next_state, self.W_state) + self.b_state)  # [B, 1]
            cum_prob_candidate = prev_cum_state_update_prob + tf.minimum(delta_state_update_prob, 1.0 - prev_cum_state_update_prob)
            cum_state_update_prob = state_update_gate * delta_state_update_prob + (1 - state_update_gate) * cum_prob_candidate

            skip_state = SkipUGRNNStateTuple(next_state, cum_state_update_prob)
            skip_output = SkipUGRNNOutputTuple(next_state, state_update_gate, delta_state_update_prob)

        return skip_output, skip_state
