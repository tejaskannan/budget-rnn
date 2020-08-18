import tensorflow as tf
from typing import Optional, List, Union, Any, Tuple

from utils.tfutils import get_activation


def dense(inputs: tf.Tensor,
          units: int,
          activation: Optional[str],
          name: str,
          use_bias: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Creates a dense, feed-forward layer with the given parameters.

    Args:
        inputs: The input tensor. Has the shape [B, ..., D]
        units: The number of output units. Denoted by K.
        activation: Optional activation function. If none, the activation is linear.
        name: Name prefix for the created trainable variables.
        use_bias: Whether to add a bias to the output.
    Returns:
        A tuple of 2 elements: (1) the transformed inputs in a [B, ..., K] tensor and (2) the transformed inputs without the activation function.
            This second entry is included for debugging purposes.
    """
    # Get the size of the input features, denoted by D
    input_units = inputs.get_shape()[-1].value

    # Get names for the trainable variables
    weight_name = '{0}-kernel'.format(name)
    bias_name = '{0}-bias'.format(name)

    # Create the weight matrix
    W = tf.get_variable(name=weight_name,
                        shape=[input_units, units],
                        initializer=tf.initializers.glorot_uniform(),
                        trainable=True)

    # Apply the given weights
    transformed = tf.matmul(inputs, W)  # [B, ..., K]

    # Add the bias if specified
    if use_bias:
        # Bias vector of size [K]
        b = tf.get_variable(name=bias_name,
                            shape=[1, units],
                            initializer=tf.initializers.random_uniform(minval=-0.7, maxval=0.7),
                            trainable=True)
        transformed = transformed + b

    pre_activation = transformed

    # Apply the activation function if specified
    activation_fn = get_activation(activation)
    if activation_fn is not None:
        transformed = activation_fn(transformed)

    return transformed, pre_activation


def mlp(inputs: tf.Tensor,
        output_size: int,
        hidden_sizes: Optional[List[int]],
        activations: Optional[Union[List[str], str]],
        name: str,
        dropout_keep_rate: float = 1.0,
        should_activate_final: bool = False,
        should_bias_final: bool = False,
        should_dropout_final: bool = False) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """
    Defines a multi-layer perceptron with the given hidden sizes, output size, and activations. This function
    is provided for convenience. It iteratively applies the `dense` function above.

    Args:
        inputs: Input tensor. Often either 2D or 3D with the leading dimension equal
            to the batch size.
        output_size: Size of each output sample. For example, if the input size
            is [B, ... D] and the output size is K, then the output tensor
            has dimensions [B, ..., K].
        hidden_sizes: Number of units in each hidden layer. None or an empty list
            signals that no hidden layers should be included.
        activations: Activation functions to apply the each layer. This can either be a list
            of different activations or a single activation function (which is applied everywhere).
        dropout_keep_rate: The dropout keep probability.
        should_activate_final: Whether to apply activations to the output layer.
        should_bias_final: Whether to apply a bias to the output layer.
        should_dropout_final: Whether to apply dropout to the final tensor.
    Returns:
        A tuple with two elements: (1) A tensor containing the inputs transformed by the MLP and (2) a list
            of the intermediate state (included for debugging purposes)
    """
    if hidden_sizes is None:
        hidden_sizes = []

    # Unpack the activation functions
    activation_fns: List[Optional[str]] = []
    if activations is None or not isinstance(activations, list):
        activation_fns = [activations] * (len(hidden_sizes) + 1)
    else:
        activation_fns = activations

    # Validate activation functions against the number of layers
    if len(activation_fns) != len(hidden_sizes) + 1:
        raise ValueError('Provided {0} for {1} layers!'.format(len(activation_fns), len(hidden_sizes) + 1))

    # Make final activation linear if specified
    if not should_activate_final:
        activation_fns[-1] = None

    states: List[tf.Tensor] = []

    # Apply hidden layers
    intermediate = inputs
    for i, (hidden_size, activation) in enumerate(zip(hidden_sizes, activation_fns[:-1])):
        intermediate, _ = dense(inputs=intermediate,
                                units=hidden_size,
                                activation=activation,
                                use_bias=True,
                                name='{0}-hidden-{1}'.format(name, i))
        intermediate = tf.nn.dropout(intermediate, keep_prob=dropout_keep_rate)
        states.append(intermediate)

    # Apply the output layer
    result, _ = dense(inputs=intermediate,
                      units=output_size,
                      activation=activation_fns[-1],
                      use_bias=should_bias_final,
                      name='{0}-output'.format(name))

    # Apply dropout to the final layer if specified
    if should_dropout_final:
        result = tf.nn.dropout(result, keep_prob=dropout_keep_rate)

    return result, states
