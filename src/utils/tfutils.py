import tensorflow as tf
from typing import Dict, Optional, List, Callable, Union, Tuple
from collections import namedtuple
from functools import partial

from utils.constants import SMALL_NUMBER


FusionLayer = namedtuple('FusionLayer', ['dense', 'bias', 'activation'])
NODES_TO_SKIP = ['initializer', 'dropout']


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
    elif fn_name == 'linear_sigmoid':
        return partial(bounded_leaky_relu, factor=0.25, size=1, shift=2, alpha=0)
    elif fn_name == 'linear_tanh':
        return partial(bounded_leaky_relu, factor=0.5, size=2, shift=1, alpha=0.0625)
    elif fn_name == 'linear':
        return None
    else:
        raise ValueError(f'Unknown activation name {fn_name}.')


def bounded_leaky_relu(x: tf.Tensor, factor: float, size: float, shift: float, alpha: float) -> tf.Tensor:
    w = tf.nn.relu(-1 * factor * x + 0.5)
    z = size * tf.nn.relu(-1 * tf.nn.relu(factor * x - 0.5) + w - 1)
    return z - size * (w - shift) - 1 + alpha * x


def get_regularizer(name: Optional[str], scale: float) -> Optional[Callable[[tf.Tensor], tf.Tensor]]:
    """
    Returns a weight regularizer with the given name and scale.
    """
    if name is None:
        return None

    name = name.lower()
    if name in ('l1', 'lasso'):
        return tf.contrib.layers.l1_regularizer(scale=scale)
    elif name == 'l2':
        return tf.contrib.layers.l2_regularizer(scale=scale)
    else:
        raise ValueError(f'Unknown regularization name: {name}')


def pool_rnn_outputs(outputs: tf.Tensor, final_state: tf.Tensor, pool_mode: str, name: str = 'pool-layer'):
    """
    Pools the outputs of an RNN using the given strategy.

    Args:
        outputs: A [B, T, D] tensor containing the RNN outputs
        final_state: A [B, D] tensor with the final RNN state
        pool_mode: Pooling strategy
        name: Name prefix for this operation
    Returns:
        A [B, D] tensor which represents an aggregation of the RNN outputs.
    """
    pool_mode = pool_mode.lower()

    if pool_mode == 'sum':
        return tf.reduce_sum(outputs, axis=-2, name=name)
    elif pool_mode == 'max':
        return tf.reduce_max(outputs, axis=-2, name=name)
    elif pool_mode == 'mean':
        return tf.reduce_mean(outputs, axis=-2, name=name)
    elif pool_mode == 'final_state':
        return final_state
    elif pool_mode == 'weighted_average':
        # [B, T, 1]
        attention_layer = tf.layers.dense(inputs=outputs,
                                          units=1,
                                          activation=get_activation('leaky_relu'),
                                          kernel_initializer=tf.initializers.glorot_uniform(),
                                          name='{0}-attention'.format(name))
        normalized_attn_weights = tf.nn.softmax(attention_layer, axis=-2, name='{0}-normalize'.format(name))  # [B, T, 1]
        scaled_outputs = tf.math.multiply(outputs, normalized_attn_weights, name='{0}-scale'.format(name))
        return tf.reduce_sum(scaled_outputs, axis=-2, name='{0}-aggregate'.format(name))  # [B, D]
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
        concat_states = tf.concat([curr_state, prev_state], axis=-1)  # [B, 2 * D]
        transform = tf.matmul(concat_states, fusion_layer.dense) + fusion_layer.bias  # [B, D]

        activation = get_activation(fusion_layer.activation)
        update_weight = activation(transform)  # [B, D]

        return update_weight * curr_state + (1.0 - update_weight) * prev_state
    else:
        raise ValueError(f'Unknown fusion mode: {mode}')

def majority_vote(logits: tf.Tensor) -> tf.Tensor:
    """
    Outputs a prediction based on a majority-voting scheme.

    Args:
        logits: A [B, T, D] tensor containing the output logits for each sequence element (T)
    Returns:
        A [B] tensor containing the predictions for each batch sample (D)
    """
    predicted_probs = tf.nn.softmax(logits, axis=-1)  # [B, T, D]
    predicted_classes = tf.argmax(predicted_probs, axis=-1)  # [B, T]

    batch_size, seq_length = tf.shape(predicted_probs)[0], tf.shape(predicted_probs)[1]
    sample_classes = tf.TensorArray(size=batch_size, dtype=tf.int32, clear_after_read=True, name='predictions')

    seq_length = tf.shape(predicted_classes)[-1]

    def body(index, predictions_array):
        sample_classes = tf.gather(predicted_classes, index)  # [T]

        label_counts = tf.bincount(tf.cast(sample_classes, dtype=tf.int32))  # [T]
        predicted_label = tf.cast(tf.argmax(label_counts), dtype=tf.int32)

        predictions_array = predictions_array.write(index=index, value=predicted_label)
        return [index + 1, predictions_array]

    def cond(index, _):
        return index < batch_size

    i = tf.constant(0)
    _, predictions_array = tf.while_loop(cond=cond, body=body,
                                         loop_vars=[i, sample_classes],
                                         parallel_iterations=1,
                                         maximum_iterations=batch_size,
                                         name='majority-while-loop')
    return predictions_array.stack()


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


def get_total_flops(node: Optional[tf.profiler.GraphNodeProto]) -> int:
    """
    Returns the total number of floating point ops in the rooted computational graph.
    """
    if node is None:
        return 0

    total_flops = node.total_float_ops
    for child in node.children:

        child_name = child.name.lower()
        should_skip = False
        for name in NODES_TO_SKIP:
            if name not in child.name.lower():
                should_skip = True
                break

        if not should_skip:
            total_flops += get_total_flops(child)

    return total_flops
