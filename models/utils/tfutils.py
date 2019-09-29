import tensorflow as tf
from typing import Dict, Optional


def get_optimizer(name: str, learning_rate: float, momentum: Optional[float]):
    momentum = momentum if momentum is not None else 0.0
    name = name.lower()

    if name == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    elif name == 'nesterov':
        return tf.keras.otpimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    elif name == 'adagrad':
        return tf.keras.optimizer.Adagrad(learning_rate=learning_rate)
    elif name == 'rmsprop':
        return tf.keras.optimizer.RMSProp(learning_rate=learning_rate)
    elif name == 'adam':
        return tf.keras.optimizer.Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f'Unknown optimizer {name}!')
