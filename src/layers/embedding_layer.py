import tensorflow as tf
from typing import Dict, Any, List, Optional

from utils.tfutils import get_activation


def embedding_layer(inputs: tf.Tensor,
                    units: int,
                    use_conv: bool,
                    dropout_keep_rate: tf.Tensor,
                    params: Dict[str, Any],
                    name_prefix: str,
                    seq_length: int,
                    input_shape: Optional[List[int]]):
    if use_conv:
        # Reshape into [B * T, H, W, C]
        conv_inputs = tf.reshape(inputs, shape=[-1, input_shape[0], input_shape[1], params['filter_channels'][0]])
 
        # Create convolution filter
        filter_heights = params['filter_heights']
        filter_widths = params['filter_widths']
        filter_strides = params['filter_strides']
        filter_channels = params['filter_channels']

        # Validate parameter lengths
        assert len(filter_heights) == len(filter_widths), 'Must provide equal number of heights and widths'
        assert len(filter_heights) == len(filter_strides), 'Must prove equal number of heights and strides'
        assert len(filter_heights) == len(filter_channels), 'Must prove equal number of heights and channels'

        # Apply convolution layers
        num_layers = len(filter_heights)
        conv_output = conv_inputs
        for layer_index in range(num_layers):
            # Get number of output channels
            output_channels = filter_channels[layer_index + 1] if layer_index < len(filter_channels) - 1 else 1

            # Apply convolution
            conv_transformed = tf.layers.conv2d(inputs=conv_output,
                                                filters=output_channels,
                                                kernel_size=(filter_heights[layer_index], filter_widths[layer_index]),
                                                strides=(filter_strides[layer_index], filter_strides[layer_index]),
                                                padding='same',
                                                activation=get_activation(params['conv_activation']),
                                                kernel_initializer=tf.glorot_uniform_initializer(),
                                                use_bias=True,
                                                name=f'{name_prefix}-filter-{layer_index}')
            # Apply dropout
            conv_output = tf.nn.dropout(conv_transformed, keep_prob=dropout_keep_rate)

        # Stack every 2d input into a 1d vector
        batch_size = tf.shape(inputs)[0]
        output_shape = conv_output.get_shape()
        inputs = tf.reshape(conv_output, [batch_size, seq_length, output_shape[1] * output_shape[2]])

    # Project down to state size, [B, T, D]
    return tf.layers.dense(inputs=inputs,
                           units=units,
                           activation=get_activation(params['dense_activation']),
                           kernel_initializer=tf.glorot_uniform_initializer(),
                           use_bias=True,
                           name=f'{name_prefix}-dense')
