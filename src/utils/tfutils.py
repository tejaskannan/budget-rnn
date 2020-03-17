import tensorflow as tf
from typing import Dict, Optional, List
from collections import namedtuple
from dpu_utils.tfutils import get_activation

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
