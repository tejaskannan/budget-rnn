"""
This file contains a collection of RNN cells retrofitted
for a sample RNN.
"""
import tensorflow as tf
from typing import Optional, Tuple, Any
from collections import namedtuple

from utils.tfutils import get_activation, apply_noise
from .cell_utils import ugrnn


BudgetOutput = namedtuple('SampleOutput', ['output', 'fusion'])


class BudgetUGRNNCell(tf.compat.v1.nn.rnn_cell.RNNCell):

    def __init__(self, units: int, activation: str, name: str, recurrent_noise: tf.Tensor):
        self._units = units
        self._activation = get_activation(activation)
        self._name = name
        self._fusion_mask = 0
        self._recurrent_noise = recurrent_noise

        # Create the trainable parameters
        self.W_transform = tf.compat.v1.get_variable(name='{0}-W-transform'.format(name),
                                                     initializer=tf.compat.v1.glorot_uniform_initializer(),
                                                     shape=[2 * units, 2 * units],
                                                     trainable=True)
        self.b_transform = tf.compat.v1.get_variable(name='{0}-b-transform'.format(name),
                                                     initializer=tf.compat.v1.glorot_uniform_initializer(),
                                                     shape=[1, 2 * units],
                                                     trainable=True)
        self.W_fusion = tf.compat.v1.get_variable(name='{0}-W-fusion'.format(name),
                                                  initializer=tf.compat.v1.glorot_uniform_initializer(),
                                                  shape=[2 * units, units],
                                                  trainable=True)
        self.b_fusion = tf.compat.v1.get_variable(name='{0}-b-fusion'.format(name),
                                                  initializer=tf.compat.v1.glorot_uniform_initializer(),
                                                  shape=[1, units],
                                                  trainable=True)

    def set_fusion_mask(self, mask_value: int):
        assert mask_value == 0 or mask_value == 1, 'Can only set the mask value to zero or 1'
        self._fusion_mask = mask_value

    @property
    def state_size(self) -> int:
        return self._units

    @property
    def output_size(self) -> BudgetOutput:
        return BudgetOutput(self._units, self._units)

    def get_initial_state(self, inputs: Optional[tf.Tensor], batch_size: Optional[int], dtype: Any) -> tf.Tensor:
        initial_state = tf.compat.v1.get_variable(name='initial-state',
                                                  initializer=tf.compat.v1.zeros_initializer(),
                                                  shape=[1, self._units],
                                                  dtype=dtype,
                                                  trainable=False)

        return tf.tile(initial_state, multiples=(batch_size, 1))  # [B, D]

    def __call__(self, inputs: tf.Tensor, state: tf.Tensor, scope=None) -> Tuple[BudgetOutput, tf.Tensor]:
        scope = scope if scope is not None else type(self).__name__

        with tf.compat.v1.variable_scope(scope):
            # Split inputs into two [B, D] tensors
            inputs, prev_state = tf.split(inputs, num_or_size_splits=2, axis=-1)

            states_concat = tf.concat([state, prev_state], axis=-1)  # [B, 2 * D]
            fusion = tf.matmul(states_concat, self.W_fusion)  # [B, D]
            fusion_gate = self._fusion_mask * (1.0 - tf.math.sigmoid(fusion + self.b_fusion))  # [B, D]
            fused_state = (1.0 - fusion_gate) * state + fusion_gate * prev_state

            # Apply the standard UGRNN update, [B, D]
            next_state = ugrnn(inputs=inputs,
                               state=fused_state,
                               W_transform=self.W_transform,
                               b_transform=self.b_transform,
                               activation=self._activation)

            # Apply regularization_noise
            next_state = apply_noise(next_state, scale=self._recurrent_noise)

        return BudgetOutput(output=next_state, fusion=fused_state), next_state
