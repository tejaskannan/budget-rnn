import tensorflow as tf
from typing import List, Optional

from utils.tfutils import get_activation, apply_noise


def conv_1d(inputs: tf.Tensor, filter_width: int, stride: int, activation: Optional[str], activation_noise: float, dropout_keep_rate: tf.Tensor, use_dropout: bool, name: str) -> tf.Tensor:
    """
    Performs a 1d convolution over the given inputs.

    Args:
        inputs: A [B, T, D] tensor of features (D) for each seq element (T) and batch sample (B)
        filter_width: The width of the convolution filter. Must be at least one.
        stride: The convolution stride. Must be at least one.
        activation: The name of the activation function. If none, then we apply a linear activation.
        activation_noise: The noise to apply to the final activations.
        dropout_keep_rate: The dropout keep rate to apply to the transformed representation.
        use_dropout: Whether to apply dropout.
        name: The name of this layer.
    Returns:
        A [B, T, D] tensor that is the result of applying the 1d convolution filter
            to the inputs.
    """
    assert filter_width >= 1, 'Must have a filter width of at least one. Got: {0}'.format(filter_width)
    assert stride >= 1, 'Must have a stride length of at least one. Got: {0}'.format(stride)

    with tf.variable_scope(name):
        # Create the (trainable) convolution filter
        num_features = inputs.get_shape()[-1]  # D
        conv_filter = tf.get_variable(shape=[filter_width, num_features, num_features],
                                      initializer=tf.glorot_uniform_initializer(),
                                      name='filter',
                                      dtype=tf.float32)

        # Create the (trainable) bias
        bias = tf.get_variable(shape=[1, 1, num_features],
                               initializer=tf.random_uniform_initializer(minval=-0.7, maxval=0.7),
                               name='bias',
                               dtype=tf.float32)

        # Apply the convolution filter, [B, T, D]
        transformed = tf.nn.conv1d(value=inputs,
                                   filters=conv_filter,
                                   stride=stride,
                                   padding='SAME',
                                   data_format='NWC')

        transformed = transformed + bias  # [B, T, D]

        # Apply the activation function, [B, T, D]
        activation_fn = get_activation(activation)
        if activation_fn is not None:
            transformed = activation_fn(transformed)

        # Apply the activation noise
        transformed = apply_noise(transformed, scale=activation_noise)

        # Apply dropout if specified, [B, T, D]
        if use_dropout:
            transformed = tf.nn.dropout(transformed, keep_prob=dropout_keep_rate)

        return transformed
