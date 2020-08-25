"""
This file contains a collection of standard RNN cells.
"""
import tensorflow as tf
from typing import Optional, Tuple, Any

from utils.tfutils import get_activation, apply_noise
from .cell_utils import ugrnn


class UGRNNCell(tf.nn.rnn_cell.RNNCell):

    def __init__(self, units: int, activation: str, name: str, recurrent_noise: tf.Tensor):
        self._units = units
        self._activation = get_activation(activation)
        self._name = name
        self._recurrent_noise = recurrent_noise

        # Create the trainable parameters
        self.W_transform = tf.get_variable(name='{0}-W-transform'.format(name),
                                           initializer=tf.glorot_uniform_initializer(),
                                           shape=[2 * units, 2 * units],
                                           trainable=True)
        self.b_transform = tf.get_variable(name='{0}-b-transform'.format(name),
                                           initializer=tf.glorot_uniform_initializer(),
                                           shape=[1, 2 * units],
                                           trainable=True)

    @property
    def state_size(self) -> int:
        return self._units

    @property
    def output_size(self) -> int:
        return self._units

    def get_initial_state(self, inputs: Optional[tf.Tensor], batch_size: Optional[int], dtype: Any) -> tf.Tensor:
        initial_state = tf.get_variable(name='initial-state',
                                        initializer=tf.zeros_initializer(),
                                        shape=[1, self._units],
                                        dtype=dtype,
                                        trainable=False)
        return tf.tile(initial_state, multiples=(batch_size, 1))  # [B, D]

    def __call__(self, inputs: tf.Tensor, state: tf.Tensor, scope=None) -> Tuple[tf.Tensor, tf.Tensor]:
        scope = scope if scope is not None else type(self).__name__

        with tf.variable_scope(scope):

            # Apply the standard UGRNN update, [B, D]
            next_state = ugrnn(inputs=inputs,
                               state=state,
                               W_transform=self.W_transform,
                               b_transform=self.b_transform,
                               activation=self._activation)

            # Apply regularization noise
            next_state = apply_noise(next_state, scale=self._recurrent_noise)

        return next_state, next_state
