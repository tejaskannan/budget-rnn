import tensorflow as tf
from typing import Optional, List, Union, Any, Tuple

from utils.tfutils import get_activation, get_regularizer, expand_to_matrix
from utils.constants import BIG_NUMBER


def pool_sequence(embeddings: tf.Tensor, pool_mode: str, name: str = 'pool-layer') -> tf.Tensor:
    """
    Args:
        embeddings: A [B, T, K] tensor
        pool_mode: Pooling strategy
    Returns:
        A [B, K] tensor containing the pooled vectors for each sequence
    """
    pool_mode = pool_mode.lower()
    if pool_mode == 'sum':
        return tf.reduce_sum(embeddings, axis=-2, name=name)
    elif pool_mode == 'average':
        return tf.reduce_mean(embeddings, axis=-2, name=name)
    elif pool_mode == 'max':
        return tf.reduce_max(embeddings, axis=-2, name=name)
    elif pool_mode == 'attention':
        attention_weights = tf.layers.dense(inputs=embeddings,
                                            units=1,
                                            activation=tf.nn.leaky_relu,
                                            use_bias=True,
                                            kernel_initializer=tf.glorot_uniform_initializer(),
                                            name='{0}-dense'.format(name))
        normalized_weights = tf.nn.softmax(attention_weights, axis=-2, name='{0}-normalize'.format(name))  # [B, T, 1]
        weighted_vectors = tf.math.multiply(embeddings, normalized_weights, name='{0}-scale'.format(name))  # [B, T, K]
        return tf.reduce_sum(weighted_vectors, axis=-2, name='{0}-aggregate'.format(name))  # [B, K]
    else:
        raise ValueError(f'Unknown pool mode {pool_mode}!')


def dense(inputs: tf.Tensor,
          units: int,
          activation: Optional[str],
          name: str,
          use_bias: bool = False,
          compression_fraction: Optional[float] = None,
          compression_seed: Optional[str] = None) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Creates a dense, feed-forward layer with the given parameters.

    Args:
        inputs: The input tensor. Has the shape [B, ..., D]
        units: The number of output units. Denoted by K.
        activation: Optional activation function. If none, the activation is linear.
        name: Name prefix for the created trainable variables.
        use_bias: Whether to add a bias to the output.
        compression_fraction: An optional number in [0, 1) that determines the number of true parameters.
            We perform compression using random weight sharing via hashing. This idea is
            from HashNets (http://proceedings.mlr.press/v37/chenc15.pdf)
        compression_seed: The prefix seed to use when hashing. Must be non-None when the compression fraction is non-None.
    Returns:
        A tuple of 2 elements: (1) the transformed inputs in a [B, ..., K] tensor and (2) the transformed inputs without the activation function.
            This second entry is included for debugging purposes.
    """
    assert compression_fraction is None or (0 <= compression_fraction and compression_fraction < 1), 'If given, the compression fraction must be in [0, 1)'

    # Get the size of the input features, denoted by D
    input_units = inputs.get_shape()[-1].value

    # Get names for the trainable variables
    weight_name = '{0}-kernel'.format(name)
    bias_name = '{0}-bias'.format(name)

    if compression_fraction is None:
        # This is a standard weight matrix
        W = tf.get_variable(name=weight_name,
                            shape=[input_units, units],
                            initializer=tf.initializers.glorot_uniform(),
                            trainable=True)
    else:
        assert compression_seed is not None, 'The seed value must be non-None when compressing weights'

        # Create the compressed weight vector (minimum of 1 weight)
        num_weights = max(int(compression_fraction * (input_units * units)), 1)
        w = tf.get_variable(name=weight_name,
                            shape=[num_weights],
                            initializer=tf.initializers.glorot_uniform(),
                            trainable=True)

        # Construct the 'virtual' weight matrix using hashing. While it is memory-inefficient to actually materialize
        # the weight matrix, it simplifies the training process. Furthermore, we are only concerned with memory constraints
        # at inference time. Thus, we just the smaller weight vector to create the full weight matrix during training.
        W = expand_to_matrix(w, size=num_weights, matrix_dims=(input_units, units), name=compression_seed)

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
        should_dropout_final: bool = False,
        regularization_name: Optional[str] = None,
        regularization_scale: float = 0.01,
        compression_fraction: Optional[float] = None,
        compression_seed: Optional[float] = None) -> Tuple[tf.Tensor, List[tf.Tensor]]:
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
        regularization_name: Name of the regularization function. None if no regularization.
        regularization_scale: Scale of regularization. Unused with the no regularization is applied.
        compression_fraction: Optional compression fraction to train network with compressed weights.
        compression_seed: Optional name of compression seed. Must be provided when compression fraction is given.
    Returns:
        A tuple with two elements: (1) A tensor containing the inputs transformed by the MLP and (2) a list
            of the intermediate state (included for debugging purposes)
    """
    if hidden_sizes is None:
        hidden_sizes = []

    if not isinstance(activations, list):
        activations = [activations] * (len(hidden_sizes) + 1)

    # Validate activation functions against the number of layers
    if len(activations) != len(hidden_sizes) + 1:
        raise ValueError(f'Provided {len(activations)} for {len(hidden_sizes) + 1} layers!')

    # Make final activation linear if specified
    if not should_activate_final:
        activations[-1] = None

    states = []

    # Apply hidden layers
    intermediate = inputs
    for i, (hidden_size, activation) in enumerate(zip(hidden_sizes, activations[:-1])):
        intermediate, _ = dense(inputs=intermediate,
                                units=hidden_size,
                                activation=activation,
                                use_bias=True,
                                name='{0}-hidden-{1}'.format(name, i),
                                compression_fraction=compression_fraction,
                                compression_seed='{0}{1}'.format(compression_seed, i))

        intermediate = tf.nn.dropout(intermediate, keep_prob=dropout_keep_rate)

        states.append(intermediate)

    # Apply the output layer
    result, _ = dense(inputs=intermediate,
                      units=output_size,
                      activation=activations[-1],
                      use_bias=should_bias_final,
                      name='{0}-output'.format(name),
                      compression_fraction=compression_fraction,
                      compression_seed=compression_seed)

    # Apply dropout to the final layer if specified
    if should_dropout_final:
        result = tf.nn.dropout(result, keep_prob=dropout_keep_rate)

    return result, states


def weighted_average(inputs: tf.Tensor,
                     mask: tf.Tensor,
                     activation: Optional[str],
                     dropout_keep_rate: tf.Tensor):
    """
    A simple attention layer which computes the weighted average
    over the final two dimensions of the given inputs.

    Args:
        inputs: A B x ... x T x K tensor
        mask: A B x ... x T binary (float) tensor to mask out certain entries
        activation: activation function to apply to get
            the activation weights
        dropout_keep_rate: Dropout to apply to attention weights
    Returns:
        A B x ... x K tensor containing the aggregated vectors
    """
    raw_weights = mlp(inputs=inputs,
                      output_size=1,
                      hidden_sizes=None,
                      activation_fns=activation,
                      name='attention-weights')

    # Mask out vectors by setting the mask values to a large negative number.
    # Applying softmax to these values will zero the masked-values out.
    attention_mask = (1.0 - tf.expand_dims(mask, axis=-1)) * -BIG_NUMBER  # B x ... x T x 1
    masked_weights = attention_mask + raw_weights

    attention_weights = tf.nn.softmax(masked_weights, axis=-1)  # B x ... x T x 1

    weighted_inputs = inputs * attention_weights  # B x ... x T x K
    return tf.reduce_sum(weighted_inputs, axis=-2)  # B x ... x K


def rnn_cell(cell_type: str,
             num_units: int,
             activation: str,
             dropout_keep_rate: float,
             name: str,
             dtype: Any,
             num_layers: Optional[int] = None,
             state_is_tuple: bool = True) -> tf.nn.rnn_cell.RNNCell:
    if num_layers is not None and num_layers <= 0:
        raise ValueError(f'The number of layers must be non-negative. Received ({num_layers}).')

    cell_type = cell_type.lower()

    def make_cell(cell_type: str, name: str):
        cell = None
        if cell_type in {'rnn', 'basic', 'vanilla'}:
            cell = tf.nn.rnn_cell.BasicRNNCell
        elif cell_type == 'gru':
            cell = tf.nn.rnn_cell.GRUCell
        elif cell_type == 'lstm':
            cell = tf.nn.rnn_cell.LSTMCell
        else:
            raise ValueError(f'Unrecognized cell type {cell_type}!')

        rnn_cell = cell(num_units=num_units,
                        activation=get_activation(activation),
                        name=name,
                        dtype=dtype)
        cell_with_dropout = tf.nn.rnn_cell.DropoutWrapper(cell=rnn_cell,
                                                          input_keep_prob=dropout_keep_rate,
                                                          state_keep_prob=dropout_keep_rate)
        return cell_with_dropout

    if num_layers is not None and num_layers > 1:
        cells = [make_cell(cell_type, name=f'{name}-{i}') for i in range(num_layers)]
        return tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)

    return make_cell(cell_type, name=name)
