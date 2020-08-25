import tensorflow as tf
from collections import namedtuple
from typing import Optional, Callable


def ugrnn(inputs: tf.Tensor, state: tf.Tensor, W_transform: tf.Tensor, b_transform: tf.Tensor, activation: Optional[Callable[[tf.Tensor], tf.Tensor]]) -> tf.Tensor:
    """
    Applies a UGRNN Cell with the given trainable parameters to the current state and input.

    Args:
        inputs: A [B, D] tensor containing the inputs at this step
        state: A [B, D] tensor containing the previous state
        W_transform: A [2*D, 2*D] (trainable) weight matrix
        b_transform: A [1, 2*D] (trainable) bias vector
        activation: The activation function for the candidate state
    Returns:
        A [B, D] tensor containing the next state
    """
    input_state_concat = tf.concat([state, inputs], axis=-1)  # [B, 2 * D]
    transformed = tf.matmul(input_state_concat, W_transform) + b_transform  # [B, 2 * D]

    update, candidate = tf.split(transformed, num_or_size_splits=2, axis=-1)  # Pair of [B, D]

    update_gate = tf.math.sigmoid(update + 1)  # [B, D]
    candidate_state = activation(candidate) if activation is not None else candidate  # [B, D]

    next_state = update_gate * state + (1.0 - update_gate) * candidate_state
    return next_state


