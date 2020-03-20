import tensorflow as tf
from typing import Dict, Optional, List, Callable, Union, Tuple
from collections import namedtuple

from .constants import SMALL_NUMBER


FusionLayer = namedtuple('FusionLayer', ['prev', 'curr', 'bias'])


def get_optimizer(name: str, learning_rate: float, learning_rate_decay: float, global_step: tf.Variable, decay_steps: int = 100000, momentum: Optional[float] = None):
    momentum = momentum if momentum is not None else 0.0
    name = name.lower()

    scheduled_learning_rate = tf.train.exponential_decay(learning_rate=learning_rate,
                                                         global_step=global_step,
                                                         decay_steps=decay_steps,
                                                         decay_rate=learning_rate_decay)
    if name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=scheduled_learning_rate)
    elif name == 'nesterov':
        return tf.train.MomentumOptimizer(learning_rate=scheduled_learning_rate, momentum=momentum)
    elif name == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate=scheduled_learning_rate)
    elif name == 'adam':
        return tf.train.AdamOptimizer(learning_rate=scheduled_learning_rate)
    else:
        raise ValueError(f'Unknown optimizer {name}!')


def get_activation(fn_name: Optional[str]) -> Optional[Callable[[tf.Tensor], tf.Tensor]]:
    """
    Returns the activation function with the given name.
    """
    if fn_name is None:
        return None

    fn_name = fn_name.lower()
    if fn_name == 'tanh':
        return tf.nn.tanh
    elif fn_name == 'relu':
        return tf.nn.relu
    elif fn_name == 'sigmoid':
        return tf.math.sigmoid
    elif fn_name == 'leaky_relu':
        return tf.nn.leaky_relu
    elif fn_name == 'elu':
        return tf.nn.elu
    elif fn_name == 'crelu':
        return tf.nn.crelu
    elif fn_name == 'linear':
        return None
    else:
        raise ValueError(f'Unknown activation name {fn_name}.')


def pool_rnn_outputs(outputs: tf.Tensor, final_state: tf.Tensor, pool_mode: str):
    """
    Pools the outputs of an RNN using the given strategy.

    Args:
        outputs: A [B, T, D] tensor containing the RNN outputs
        final_state: A [B, D] tensor with the final RNN state
        pool_mode: Pooling strategy
    Returns:
        A [B, D] tensor which represents an aggregation of the RNN outputs.
    """
    pool_mode = pool_mode.lower()

    if pool_mode == 'sum':
        return tf.reduce_sum(outputs, axis=-2)
    elif pool_mode == 'max':
        return tf.reduce_max(outputs, axis=-2)
    elif pool_mode == 'mean':
        return tf.reduce_mean(outputs, axis=-2)
    elif pool_mode == 'final_state':
        return final_state
    elif pool_mode == 'weighted_average':
        # [B, T, 1]
        attention_layer = tf.layers.dense(inputs=outputs,
                                          units=1,
                                          activation=get_activation('leaky_relu'),
                                          kernel_initializer=tf.initializers.glorot_uniform(),
                                          name='attention-layer')
        normalized_attn_weights = tf.nn.softmax(attention_layer, axis=-2)  # [B, T, 1]
        return tf.reduce_sum(outputs * normalized_attn_weights, axis=-2)  # [B, D]
    else:
        raise ValueError(f'Unknown pool mode {pool_mode}.')


def fuse_states(curr_state: tf.Tensor, prev_state: Optional[tf.Tensor], fusion_layer: Optional[FusionLayer], mode: Optional[str]) -> tf.Tensor:
    """
    Combines the provided states using the specified strategy.

    Args:
        curr_state: A [B, D] tensor of the current state
        prev_state: An optional [B, D] tensor of previous state
        fusion_layer: Optional trainable variables for the fusion layer. Only use when mode = 'gate'
        mode: The fusion strategy. If None is given, the strategy is an identity.
    Returns:
        A [B, D] tensor that represents the fused state
    """
    mode = mode.lower() if mode is not None else None

    if mode is None or mode in ('identity', 'none') or prev_state is None:
        return curr_state
    elif mode == 'sum':
        return curr_state + prev_state
    elif mode in ('sum-tanh', 'sum_tanh'):
        return tf.nn.tanh(curr_state + prev_state)
    elif mode in ('avg', 'average'):
        return (curr_state + prev_state) / 2.0
    elif mode in ('max', 'max-pool', 'max_pool'):
        concat = tf.concat([tf.expand_dims(curr_state, axis=-1), tf.expand_dims(prev_state, axis=-1)], axis=-1)  # [B, D, 2]
        return tf.reduce_max(concat, axis=-1)  # [B, D]
    elif mode in ('gate', 'gate_layer', 'gate-layer'):
        curr_transform = fusion_layer.curr(curr_state)  # [B, D]
        prev_transform = fusion_layer.prev(prev_state)  # [B, D]
        update_weight = tf.math.sigmoid(prev_transform + curr_transform + fusion_layer.bias)
        return update_weight * curr_state + (1.0 - update_weight) * prev_state
    else:
        raise ValueError(f'Unknown fusion mode: {mode}')


def variables_for_loss_op(variables: List[tf.Variable], loss_op: str) -> List[tf.Variable]:
    """
    Gets all variables that have a gradient with respect to the given loss operation.

    Args:
        variables: List of trainable variables
        loss_op: Operation to compute gradient for
    Returns:
        A list of all variables with an existing gradient
    """
    gradients = tf.gradients(loss_op, variables)
    return [v for g, v in zip(gradients, variables) if g is not None]


def tf_precision(predictions: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """
    Computes precision of the given predictions.

    Args:
        predictions: A [B, 1] tensor of model predictions.
        labels: A [B, 1] tensor of expected labels.
    Returns:
        A scalar tensor containing batch-wise precision.
    """
    true_positives = tf.reduce_sum(predictions * labels)
    false_positives = tf.reduce_sum(predictions * (1.0 - labels))

    return tf.where(tf.abs(true_positives + false_positives) < SMALL_NUMBER,
                    x=1.0,
                    y=true_positives / (true_positives + false_positives))


def tf_recall(predictions: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """
    Computes recall of the given predictions.

    Args:
        predictions: A [B, 1] tensor of model predictions.
        labels: A [B, 1] tensor of expected labels.
    Returns:
        A scalar tensor containing batch-wise recall.
    """
    true_positives = tf.reduce_sum(predictions * labels)
    false_negatives = tf.reduce_sum((1.0 - predictions) * labels)

    return tf.where(tf.abs(true_positives + false_negatives) < SMALL_NUMBER,
                    x=1.0,
                    y=true_positives / (true_positives + false_negatives))


def tf_f1_score(predictions: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """
    Computes the F1 score (harmonic mean of precision and recall) of the given predictions.

    Args:
        predictions: A [B, 1] tensor of model predictions.
        labels: A [B, 1] tensor of expected labels.
    Returns:
        A scalar tensor containing the batch-wise F1 score.
    """
    precision = tf_precision(predictions, labels)
    recall = tf_recall(predictions, labels)

    return 2 * (precision * recall) / (precision + recall + SMALL_NUMBER)


def tf_rnn_cell(cell_type: str, num_units: int, activation: str, layers: int, dropout_keep_rate: tf.Tensor, name_prefix: Optional[str]) -> tf.nn.rnn_cell.MultiRNNCell:

    def make_cell(cell_type: str, num_units: int, activation: str, name: str):
        if cell_type == 'vanilla':
            return tf.nn.rnn_cell.BasicRNNCell(num_units=num_units,
                                            activation=get_activation(activation),
                                            name=name)
        elif cell_type == 'gru':
            return tf.nn.rnn_cell.GRUCell(num_units=num_units,
                                       activation=get_activation(activation),
                                       kernel_initializer=tf.glorot_uniform_initializer(),
                                       name=name)
        elif cell_type == 'lstm':
            return tf.nn.rnn_cell.LSTMCell(num_units=num_units,
                                        activation=get_activation(activation),
                                        initializer=tf.glorot_uniform_initializer(),
                                        name=name)
        raise ValueError(f'Unknown cell type: {cell_type}')

    cell_type = cell_type.lower()
    cells: List[tf.rnn_cell.RNNCell] = []
    name_prefix = f'{name_prefix}-cell' if name_prefix is not None else 'cell'
    for i in range(layers):
        name = f'{name_prefix}-{i}'
        cell = make_cell(cell_type, num_units, activation, name)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell,
                                             input_keep_prob=dropout_keep_rate,
                                             state_keep_prob=dropout_keep_rate,
                                             output_keep_prob=dropout_keep_rate)
        cells.append(cell)

    return tf.nn.rnn_cell.MultiRNNCell(cells)

def get_rnn_state(state: Union[tf.Tensor, tf.nn.rnn_cell.LSTMStateTuple, Tuple[tf.Tensor, ...], Tuple[tf.nn.rnn_cell.LSTMStateTuple, ...]]) -> tf.Tensor:
    if isinstance(state, tf.nn.rnn_cell.LSTMStateTuple):
        return state.c
    if isinstance(state, tuple):
        states: List[tf.Tensor] = []
        for st in state:
            if isinstance(st, tf.nn.rnn_cell.LSTMStateTuple):
                states.append(st.c)
            else:
                states.append(st)
        return tf.concat(states, axis=-1)
    else:
        return state
