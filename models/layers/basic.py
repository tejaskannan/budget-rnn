import tensorflow as tf
from dpu_utils.tfutils import get_activation
from typing import Optional, List, Union


def mlp(inputs: tf.Tensor,
        output_size: int,
        hidden_sizes: Optional[List[int]],
        activations: Optional[Union[List[str], str]],
        name: str,
        dropout_keep_rate: float = 1.0,
        should_activate_final: bool = False,
        should_bias_final: bool = False,
        should_dropout_final: bool = False) -> tf.Tensor:
    """
    Defines a multi-layer perceptron with the given hidden sizes, output size, and activations.

    Args:
        inputs: Input tensor. Often either 2D or 3D with the leading dimension equal
            to the batch size.
        output_size: Size of each output sample. For example, if the input size
            is B x T x D and the output size is K, then the output tensor
            has dimensions B x T x K.
        hidden_sizes: Number of units in each hidden layer. None or an empty list
            signals that no hidden layers should be included.
        activations: Activation functions to apply the each layer. This can either be a list
            of different activations or a single activation function (which is applied everywhere).
        dropout_keep_rate: The dropout keep probability.
        should_activate_final: Whether to apply activations to the output layer.
        should_bias_final: Whether to apply a bias to the output layer.
        should_dropout_final: Whether to apply dropout to the final tensor.
    Returns:
        A tensor containing the inputs transformed by the MLP.
    """
    if hidden_sizes is None:
        hidden_sizes = []

    # Convert activation function names to the proper functions
    if isinstance(activations, list):
        activation_fns = [get_activation(a) for a in activations]
    else:
        activation_fns = [get_activation(activations) for _ in range(0, len(hidden_sizes) + 1)]

    # Validate activation functions against the number of layers
    if len(activation_fns) != len(hidden_sizes) + 1:
        raise ValueError(f'Provided {len(activation_fns)} for {len(hidden_sizes) + 1} layers!')

    # Make final activation linear if specified
    if not should_activate_final:
        activation_fns[-1] = None

    # Apply hidden layers
    intermediate = inputs
    for i, (hidden_size, activation) in enumerate(zip(hidden_sizes, activation_fns[:-1])):
        intermediate = tf.layers.dense(inputs=intermediate,
                                       units=hidden_size,
                                       activation=activation,
                                       kernel_initializer=tf.initializers.glorot_uniform(),
                                       use_bias=True,
                                       name=f'{name}-hidden-{i}')
        intermediate = tf.nn.dropout(intermediate, keep_prob=dropout_keep_rate)

    # Apply the output layer
    result = tf.layers.dense(inputs=intermediate,
                             units=output_size,
                             activation=activation_fns[-1],
                             kernel_initializer=tf.initializers.glorot_uniform(),
                             use_bias=should_bias_final,
                             name=f'{name}-output')

    # Apply dropout to the final layer if specified
    if should_dropout_final:
        result = tf.nn.dropout(result, keep_prob=dropout_keep_rate)

    return result
