import tensorflow as tf
from typing import Dict, Optional


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
