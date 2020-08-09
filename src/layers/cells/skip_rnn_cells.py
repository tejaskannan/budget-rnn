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
from utils.constants import SMALL_NUMBER


SkipGRUStateTuple = namedtuple('SkipGRUStateTuple', ['state', 'cumulative_state_update'])
SkipUGRNNStateTuple = namedtuple('SkipUGRNNStateTuple', ['state', 'cumulative_state_update'])
SoftSkipUGRNNStateTuple = namedtuple('SoftSkipUGRNNStateTuple', ['state', 'cumulative_state_update'])
InputSkipUGRNNStateTuple = namedtuple('InputSkipUGRNNStateTuple', ['state', 'prev_input'])

SkipGRUOutputTuple = namedtuple('SkipGRUOutputTuple', ['output', 'state_update_gate'])
SkipUGRNNOutputTuple = namedtuple('SkipUGRNNOutputTuple', ['output', 'state_update_gate'])
SoftSkipUGRNNOutputTuple = namedtuple('SoftSkipUGRNNOutputTuple', ['output', 'state_update_gate'])
InputSkipUGRNNOutputTuple = namedtuple('InputSkipUGRNNOutputTuple', ['output', 'input_weight'])


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

    def get_initial_state(self, inputs: Optional[tf.Tensor], batch_size: Optional[int], dtype: Any) -> SkipUGRNNStateTuple:
        """
        Creates an initial state by setting the hidden state to zero and the update probability to 1.
        """
        initial_state = tf.get_variable(name='initial-hidden-state',
                                        initializer=tf.zeros_initializer(),
                                        shape=[1, self._units],
                                        dtype=dtype,
                                        trainable=False)
        initial_state_update_prob = tf.get_variable(name='initial-state-update-prob',
                                                    initializer=tf.ones_initializer(),
                                                    shape=[1, 1],
                                                    dtype=dtype,
                                                    trainable=False)

        # We tile the initial states across the entire batch
        return SkipUGRNNStateTuple(state=tf.tile(initial_state, multiples=(batch_size, 1)),
                                   cumulative_state_update=tf.tile(initial_state_update_prob, multiples=(batch_size, 1)))

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
            delta_state_update_prob = tf.math.sigmoid(tf.matmul(next_state, self.W_state) + self.b_state)  # [B, 1]
            cum_prob_candidate = prev_cum_state_update_prob + tf.minimum(delta_state_update_prob, 1.0 - prev_cum_state_update_prob)
            cum_state_update_prob = state_update_gate * delta_state_update_prob + (1 - state_update_gate) * cum_prob_candidate

            skip_state = SkipUGRNNStateTuple(next_state, cum_state_update_prob)
            skip_output = SkipUGRNNOutputTuple(next_state, state_update_gate)

        return skip_output, skip_state


class RandomSkipUGRNNCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, units: int, activation: str, state_keep_prob: float, name: str):
        self._units = units
        self._activation = get_activation(activation)
        self._state_keep_prob = state_keep_prob

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
    def state_size(self) -> int:
        return self._units

    @property
    def output_size(self) -> int:
        return self._units

    def get_initial_state(self, inputs: Optional[tf.Tensor], batch_size: Optional[int], dtype: Any) -> SkipUGRNNStateTuple:
        """
        Creates an initial state by setting the hidden state to zero and the update probability to 1.
        """
        initial_state = tf.get_variable(name='initial-hidden-state',
                                        initializer=tf.zeros_initializer(),
                                        shape=[1, self._units],
                                        dtype=dtype,
                                        trainable=False)
        return initial_state

    def __call__(self, inputs: tf.Tensor, state: tf.Tensor, scope=None) -> Tuple[SoftSkipUGRNNOutputTuple, SoftSkipUGRNNStateTuple]:
        scope = scope if scope is not None else type(self).__name__
        with tf.variable_scope(scope):
            # Compute the standard UGRNN Cell with the added state update multiplier
            update_state = tf.matmul(prev_state, self.W_update)
            update_input = tf.matmul(inputs, self.U_update)
            update = tf.math.sigmoid(update_state + update_input + self.b_update + 1)

            candidate_state = tf.matmul(prev_state, self.W)
            candidate_input = tf.matmul(inputs, self.U)
            candidate = self._activation(candidate_state + candidate_input + self.b)

            next_state = update * candidate + (1 - update) * prev_state

            # Make Updates Randomly
            random_sample = tf.uniform(shape=(tf.shape(inputs)[0], 1), minval=0.0, maxval=1.0)
            update_mask = tf.cast(random_sample < self._state_keep_prob, dtype=tf.float32)
            next_state = update_mask * next_state + (1.0 - update_mask) * state

        return next_state, next_state


class SoftSkipUGRNNCell(tf.nn.rnn_cell.RNNCell):

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

        self.W_state_combine = tf.get_variable(name='{0}-W-state-combine-kernel'.format(name),
                                               initializer=tf.glorot_uniform_initializer(),
                                               shape=[2, 1],
                                               trainable=True)
        #self.b_state_combine = tf.get_variable(name='{0}-b-state-combine-bias'.format(name),
        #                                       initializer=tf.glorot_uniform_initializer(),
        #                                       shape=[1, 1],
        #                                       trainable=True)


    @property
    def state_size(self) -> SoftSkipUGRNNStateTuple:
        return SoftSkipUGRNNStateTuple(self._units, 1)

    @property
    def output_size(self) -> SoftSkipUGRNNOutputTuple:
        return SoftSkipUGRNNOutputTuple(self._units, 1)

    def get_initial_state(self, inputs: Optional[tf.Tensor], batch_size: Optional[int], dtype: Any) -> SkipUGRNNStateTuple:
        """
        Creates an initial state by setting the hidden state to zero and the update probability to 1.
        """
        initial_state = tf.get_variable(name='initial-hidden-state',
                                        initializer=tf.zeros_initializer(),
                                        shape=[1, self._units],
                                        dtype=dtype,
                                        trainable=False)

        initial_state_update = tf.get_variable(name='initial-state-update',
                                               initializer=tf.ones_initializer(),
                                               shape=[1, 1],
                                               dtype=dtype,
                                               trainable=False)

        # We tile the initial states across the entire batch
        return SoftSkipUGRNNStateTuple(state=tf.tile(initial_state, multiples=(batch_size, 1)),
                                       cumulative_state_update=tf.tile(initial_state_update, multiples=(batch_size, 1)))

    def __call__(self, inputs: tf.Tensor, state: SoftSkipUGRNNStateTuple, scope=None) -> Tuple[SoftSkipUGRNNOutputTuple, SoftSkipUGRNNStateTuple]:
        # Unpack the previous state
        prev_state, state_update_gate = state

        scope = scope if scope is not None else type(self).__name__
        with tf.variable_scope(scope):
            # Compute the standard UGRNN Cell with the added state update multiplier
            update_state = tf.matmul(prev_state, self.W_update)
            update_input = tf.matmul(inputs, self.U_update)
            update = tf.math.sigmoid(update_state + update_input + self.b_update + 1)
            update = update * state_update_gate

            candidate_state = tf.matmul(prev_state, self.W)
            candidate_input = tf.matmul(inputs, self.U)
            candidate = self._activation(candidate_state + candidate_input + self.b)

            next_state = update * candidate + (1 - update) * prev_state

            # Compute the next state update weight
           # delta_state_update = tf.math.sigmoid(tf.matmul(next_state, self.W_state) + self.b_state)  # [B, 1]
           # candidate_state_update = cum_state_update_gate + tf.minimum(delta_state_update, 1 - cum_state_update_gate)
           # next_cum_state_update = cum_state_update_gate * delta_state_update + (1 - cum_state_update_gate) * candidate_state_update

            candidate_state_update = tf.math.sigmoid(tf.matmul(next_state, self.W_state) + self.b_state)  # [B, 1]
            state_update_concat = tf.concat([state_update_gate, candidate_state_update], axis=-1)  # [B, 2]
            next_state_update_gate = tf.math.sigmoid(tf.matmul(state_update_concat, self.W_state_combine))

            skip_state = SoftSkipUGRNNStateTuple(next_state, next_state_update_gate)
            skip_output = SoftSkipUGRNNOutputTuple(next_state, next_state_update_gate)

        return skip_output, skip_state


class InputSkipUGRNNCell(tf.nn.rnn_cell.RNNCell):

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
    def state_size(self) -> InputSkipUGRNNStateTuple:
        return InputSkipUGRNNStateTuple(self._units, self._units)

    @property
    def output_size(self) -> InputSkipUGRNNOutputTuple:
        return InputSkipUGRNNOutputTuple(self._units, 1)

    def get_initial_state(self, inputs: Optional[tf.Tensor], batch_size: Optional[int], dtype: Any) -> tf.Tensor:
        """
        Creates an initial state by setting the hidden state to zero and the update probability to 1.
        """
        initial_state = tf.get_variable(name='initial-hidden-state',
                                        initializer=tf.zeros_initializer(),
                                        shape=[1, self._units],
                                        dtype=dtype,
                                        trainable=False)
        initial_input = tf.get_variable(name='initial-input',
                                        initializer=tf.zeros_initializer(),
                                        shape=[1, self._units],
                                        dtype=dtype,
                                        trainable=False)

        return InputSkipUGRNNStateTuple(state=tf.tile(initial_state, multiples=(batch_size, 1)),
                                        prev_input=tf.tile(initial_input, multiples=(batch_size, 1)))

    def __call__(self, inputs: tf.Tensor, state: tf.Tensor, scope: str = None) -> Tuple[InputSkipUGRNNOutputTuple, InputSkipUGRNNStateTuple]:
        prev_state, prev_inputs = state

        scope = scope if scope is not None else type(self).__name__
        with tf.variable_scope(scope):
            # Compute the input scaling weight
            input_weight = tf.math.sigmoid(tf.matmul(prev_state, self.W_state) + self.b_state)  # [B, 1]

            # Apply the input scaling weight
            cell_inputs = input_weight * inputs + (1 - input_weight) * prev_inputs

            # Compute the standard UGRNN Cell with the added state update multiplier
            update_state = tf.matmul(prev_state, self.W_update)
            update_input = tf.matmul(cell_inputs, self.U_update)
            update = tf.math.sigmoid(update_state + update_input + self.b_update + 1)

            candidate_state = tf.matmul(prev_state, self.W)
            candidate_input = tf.matmul(cell_inputs, self.U)
            candidate = self._activation(candidate_state + candidate_input + self.b)

            next_state = update * candidate + (1 - update) * prev_state

            output = InputSkipUGRNNOutputTuple(output=next_state, input_weight=input_weight)
            state = InputSkipUGRNNStateTuple(state=next_state, prev_input=cell_inputs)

        return output, state