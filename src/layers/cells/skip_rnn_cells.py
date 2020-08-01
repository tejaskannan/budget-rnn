"""
This file implements various Skip RNN Cells. The classes here
are inspired by the following repository and associated paper:

Paper: https://arxiv.org/abs/1708.06834
Repo: https://github.com/imatge-upc/skiprnn-2017-telecombcn/blob/master/src/rnn_cells/skip_rnn_cells.py.
"""
import tensorflow as tf
from collections import namedtuple
from typing import Optional, Any, Tuple

from utils.tfutils import get_activation


SkipGRUStateTuple = namedtuple('SkipGRUStateTuple', ['state', 'cumulative_state_update'])
SkipUGRNNStateTuple = namedtuple('SkipUGRNNStateTuple', ['state', 'cumulative_state_update'])
SkipGRUOutputTuple = namedtuple('SkipGRUOutputTuple', ['output', 'state_update_gate'])
SkipUGRNNOutputTuple = namedtuple('SkipUGRNNOutputTuple', ['output', 'state_update_gate'])


def binarize(x: tf.Tensor, name: str = 'binarize') -> tf.Tensor:
    """
    Maps the values in the given tensor to {0, 1} using a rounding function. This function
    assigns the gradient to be the identity.
    """
    g = tf.get_default_graph()

    with g.gradient_override_map({'Round': 'Identity'}):
        return tf.round(x, name=name)


class SkipUGRNNCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, units: int, activation: str, name: str):
        self._units = units
        self._activation = get_activation(activation)

        # Make the trainable variables for this cell
        self.W_update = tf.get_variable(name='{0}-W-update-kernel'.format(name),
                                        initializer=tf.glorot_uniform_initializer(),
                                        shape=[units, units],
                                        trainable=True)
        self.U_update = tf.get_variable(name='{0}-U-update-kernel'.format(name),
                                        initializer=tf.glorot_uniform_initializer(),
                                        shape=[units, units],
                                        trainable=True)
        self.b_update = tf.get_variable(name='{0}-b-update-bias'.format(name),
                                        initializer=tf.glorot_uniform_initializer(),
                                        shape=[1, units],
                                        trainable=True)

        self.W = tf.get_variable(name='{0}-W-kernel'.format(name),
                                 initializer=tf.glorot_uniform_initializer(),
                                 shape=[units, units],
                                 trainable=True)
        self.U = tf.get_variable(name='{0}-U-kernel'.format(name),
                                 initializer=tf.glorot_uniform_initializer(),
                                 shape=[units, units],
                                 trainable=True)
        self.b = tf.get_variable(name='{0}-b-bias'.format(name),
                                 initializer=tf.glorot_uniform_initializer(),
                                 shape=[1, units],
                                 trainable=True)

        self.W_state = tf.get_variable(name='{0}-W-state-kernel'.format(name),
                                       initializer=tf.glorot_uniform_initializer(),
                                       shape=[units, 1],
                                       trainable=True)
        self.b_state = tf.get_variable(name='{0}-b-state-bias'.format(name),
                                       initializer=tf.glorot_uniform_initializer(),
                                       shape=[1, 1],
                                       trainable=True)

    @property
    def state_size(self) -> SkipUGRNNStateTuple:
        return SkipUGRNNStateTuple(self._units, 1)
   
    @property
    def output_size(self) -> SkipUGRNNOutputTuple:
        return SkipUGRNNOutputTuple(self._units, 1)

    def get_intial_state(self, inputs: Optional[tf.Tensor], batch_size: Optional[int], dtype: Any) -> SkipUGRNNStateTuple:
        """
        Creates an initial state by setting the hidden state to zero and the update probability to 1.
        """
        initial_state = tf.get_variable(name='initial-hidden-state',
                                        initial_state=tf.zeros_intializer(),
                                        shape=[batch_size, self._units],
                                        dtype=dtype,
                                        trainable=False)
        initial_state_update_prob = tf.get_variable(name='initial-state-update-prob',
                                                    initial_state=tf.ones_initializer(),
                                                    shape=[batch_size, 1],
                                                    dtype=dtype,
                                                    trainable=False)
        return SkipUGRNNStateTuple(initial_state, initial_state_update_prob)


    def __call__(self, inputs: tf.Tensor, state: SkipUGRNNStateTuple, scope=None) -> Tuple[SkipUGRNNStateTuple, SkipUGRNNOutputTuple]:
        # Unpack the previous state
        prev_state, prev_cum_state_update_prob = state

        scope = scope if scope is not None else type(self).__name__
        with tf.variable_scope(scope):
            # Compute the standard UGRNN Cell
            update_state = tf.matmul(prev_state, self.W_update)
            update_input = tf.matmul(inputs, self.U_update)
            update = tf.math.sigmoid(update_state + update_input + self.b_update + 1)
    
            candidate_state = tf.matmul(prev_state, self.W)
            candidate_input = tf.matmul(inputs, self.U)
            candidate = self._activation(candidate_state + candidate_input + self.b)

            next_cell_state = update * candidate + (1 - update) * prev_state

            # Apply the state update gate. This is the Skip portion.
            # We first compute the state update gate. This is a binary version of the cumulative state update prob.
            state_update_gate = binarize(prev_cum_state_update_prob)  # A binary variable

            # Apply the binary state update gate to get the next state
            next_state = state_update_gate * next_cell_state + (1 - state_update_gate) * prev_state

            # Compute the next state update probability
            delta_state_update_prob = tf.math.sigmoid(tf.matmul(next_state, self.W_state) + self.b_state)
            cum_prob_candidate = prev_cum_state_update_prob + tf.minimum(delta_state_update_prob, 1.0 - prev_cum_state_update_prob)
            cum_state_update_prob = state_update_gate * delta_state_update_prob + (1 - state_update_gate) * cum_prob_candidate

            skip_state = SkipUGRNNStateTuple(next_state, cum_state_update_prob)
            skip_output = SkipUGRNNOutputTuple(next_state, state_update_gate)

        return skip_output, skip_state
