import tensorflow as tf
from typing import Dict, Optional, List
from dpu_utils.tfutils import get_activation


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
    if pool_mode == 'sum':
        return tf.reduce_sum(outputs, axis=-2)
    elif pool_mode == 'max':
        return tf.reduce_max(outputs, axis=-2)
    elif pool_mode == 'mean':
        return tf.reduce_mean(outputs, axis=-2)
    elif pool_mode == 'final_state':
        return final_state
    elif pool_mode == 'weighted_average':
        # B x T x 1
        attention_layer = tf.layers.dense(inputs=outputs,
                                          units=1,
                                          activation=get_activation('leaky_relu'),
                                          kernel_initializer=tf.initializers.glorot_uniform(),
                                          name='attention-layer')
        normalized_attn_weights = tf.nn.softmax(attention_layer, axis=-2)  # [B, T, 1]
        return tf.reduce_sum(outputs * normalized_attn_weights, axis=-2)  # [B, D]
    else:
        raise ValueError(f'Unknown pool mode {pool_mode}.')


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
