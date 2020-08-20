"""
This file contains a collection of RNN cells retrofitted
for a sample RNN.
"""
import tensorflow as tf
from typing import Optional, Tuple, Any
from collections import namedtuple

from utils.tfutils import get_activation


SampleState = namedtuple('SampleState', ['state', 'should_apply_fusion'])


class SampleUGRNNCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, units: int, activation: str, name: str):
        self._units = units
        self._activation = get_activation(activation)
        self._name = name

        # Create the trainable parameters
        self._W_candidate = tf.get_variable(name='{0}-W-candidate'.format(name),
                                            initializer=tf.glorot_uniform_initializer(),
                                            shape=[2 * units, units],
                                            trainable=True)
        self._b_candidate = tf.get_variable(name='{0}-b-candidate'.format(name),
                                            initializer=tf.glorot_uniform_initializer(),
                                            shape=[1, units],
                                            trainable=True)
        self._W_update = tf.get_variable(name='{0}-W-update'.format(name),
                                         initializer=tf.glorot_uniform_initializer(),
                                         shape=[2 * units, units],
                                         trainable=True)
        self._b_update = tf.get_variable(name='{0}-b-update'.format(name),
                                         initializer=tf.glorot_uniform_initializer(),
                                         shape=[1, units],
                                         trainable=True)
        self._W_fusion = tf.get_variable(name='{0}-W-fusion'.format(name),
                                         initializer=tf.glorot_uniform_initializer(),
                                         shape=[2 * units, units],
                                         trainable=True)
        self._b_fusion = tf.get_variable(name='{0}-b-fusion'.format(name),
                                         initializer=tf.glorot_uniform_initializer(),
                                         shape=[1, units],
                                         trainable=True)

    @property
    def state_size(self) -> SampleState:
        return SampleState(self._units, 1)

    @property
    def output_size(self) -> int:
        return self._units

    def get_initial_state(self, inputs: Optional[tf.Tensor], batch_size: Optional[int], should_apply_fusion: bool, dtype: Any) -> tf.Tensor:
        initial_state = tf.get_variable(name='initial-state',
                                        initializer=tf.zeros_initializer(),
                                        shape=[1, self._units],
                                        dtype=dtype,
                                        trainable=False)
        fusion_mask = tf.get_variable(name='fusion-mask',
                                      initializer=tf.ones_initializer() if should_apply_fusion else tf.zeros_initializer(),
                                      shape=[1, 1],
                                      dtype=dtype,
                                      trainable=False)

        return SampleState(state=tf.tile(initial_state, multiples=(batch_size, 1)),  # [B, D]
                           should_apply_fusion=tf.tile(fusion_mask, multiples=(batch_size, 1)))

    def __call__(self, inputs: tf.Tensor, state: SampleState, scope=None) -> Tuple[tf.Tensor, SampleState]:
        scope = scope if scope is not None else type(self).__name__

        state, fusion_mask = state

        with tf.variable_scope(scope):
            # Split inputs into two [B, D] tensors
            inputs, prev_state = tf.split(inputs, num_or_size_splits=2, axis=-1)

            states_concat = tf.concat([state, prev_state], axis=-1)  # [B, 2 * D]
            fusion = tf.matmul(states_concat, self._W_fusion)  # [B, D]
            fusion_gate = fusion_mask * (1.0 - tf.math.sigmoid(fusion + self._b_fusion))  # [B, D]
            fused_state = (1.0 - fusion_gate) * state + fusion_gate * prev_state

            input_state_concat = tf.concat([fused_state, inputs], axis=-1)  # [B, 2 * D]
            update = tf.matmul(input_state_concat, self._W_update)  # [B, D]
            update_gate = tf.math.sigmoid(update + self._b_update + 1)  # [B, D]

            candidate = tf.matmul(input_state_concat, self._W_candidate)  # [B, D]
            candidate_state = self._activation(candidate + self._b_candidate)  # [B, D]

            next_state = update_gate * fused_state + (1.0 - update_gate) * candidate_state

        return next_state, SampleState(next_state, fusion_mask)
