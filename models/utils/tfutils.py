import tensorflow as tf
from typing import Dict, Optional
from dpu_utils.tfutils import get_activation


def get_optimizer(name: str, learning_rate: float, momentum: Optional[float] = None):
    momentum = momentum if momentum is not None else 0.0
    name = name.lower()

    if name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    elif name == 'nesterov':
        return tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    elif name == 'adagrad':
        return tf.train.AdagradOptimizer(learning_rate=learning_rate)
    elif name == 'adam':
        return tf.train.AdamOptimizer(learning_rate=learning_rate)
    else:
        raise ValueError(f'Unknown optimizer {name}!')


def pool_rnn_outputs(outputs: tf.Tensor, final_state: tf.Tensor, pool_mode: str):
    """
    Pools the outputs of an RNN using the given strategy.

    Args:
        outputs: A B x T x D tensor containing the RNN outputs
        final_state: A B x D tensor with the final RNN state
        pool_mode: Pooling strategy
    Returns:
        A B x D tensor which represents an aggregation of the RNN outputs.
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
                                          activation=get_activation('tanh'),
                                          kernel_initializer=tf.initializers.glorot_uniform(),
                                          name='attention-layer')
        normalized_attn_weights = tf.nn.softmax(attention_layer, axis=-2)  # B x T x 1
        return outputs * normalized_attn_weights  # B x T x D
    else:
        raise ValueError(f'Unknown pool mode {pool_mode}.')
