import tensorflow as tf
import numpy as np
from typing import Optional

from .constants import SMALL_NUMBER, ONE_HALF


BETA = 10.0


def get_loss_weights(n: int, mode: Optional[str]) -> np.ndarray:
    """
    Returns the loss weights for the given mode. The weights are normalized
    to sum to one.

    Args:
        n: Number of weights
        mode: Optional name of the loss mode. Must be one of None, 'linear', or 'quadratic'.
            If None, then the weighting is even.
    Returns:
        An array of normalized weights for each output.
    """
    # Construct the weights
    if mode is None or not isinstance(mode, str):
        weights = np.ones(shape=(n, ))
    elif mode.lower() == 'linear':
        weights = np.linspace(start=1, stop=n, num=n, endpoint=True)
    elif mode.lower() == 'quadratic':
        weights = np.linspace(start=1, stop=n, num=n, endpoint=True)
        weights = np.square(weights)
    else:
        # The default behavior is even weights. We do this for backwards compatibility.
        weights = np.ones(shape=(n, ))

    # Normalize to a simplex
    weights = weights / np.sum(weights, keepdims=True)

    return weights


def get_temperate_loss_weight(start_weight: float, end_weight: float, step: int, max_steps: int) -> float:
    if step >= max_steps or end_weight <= start_weight:
        return max(end_weight, 0)

    k = np.exp((1.0 / max_steps) * np.log(end_weight / start_weight))
    return np.power(k, step) * start_weight



def binary_classification_loss(predicted_probs: tf.Tensor,
                               predictions: tf.Tensor, 
                               labels: tf.Tensor, 
                               pos_weight: float, 
                               neg_weight: float) -> tf.Tensor:
        """
        Computes a weighted binary cross entropy loss to bias towards specific types of errors.

        Args:
            predicted_probs: A [B, 1] tensor of predicted probabilities
            predictions: A [B, 1] tensor of predicted labels
            labels: A [B, 1] tensor of expected labels
            pos_weight: Weight to multiply on false positives
            neg_weight: Weight to multiply on false negatives
        """
        log_probs = -tf.log(predicted_probs)
        log_opp_probs = -tf.log(1.0 - predicted_probs)
        sample_loss = tf.where((1.0 - tf.abs(predictions - labels)) < SMALL_NUMBER,
                               x=neg_weight * labels * log_probs + pos_weight * (1.0 - labels) * log_opp_probs,  # False negative or False Positive
                               y=labels * log_probs + (1.0 - labels) * log_opp_probs) 
        return tf.reduce_mean(sample_loss)


def f1_score_loss(predicted_probs: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """
    Computes a loss function based on F1 scores (harmonic mean of precision an recall).

    Args:
        predicted_probs: A [B, L] tensor of predicted probabilities
        labels: A [B, 1] tensor of expected labels
    Returns:
        A tensor of sample-wise losses
    """
    # Apply a sharpened sigmoid function to approximate the threshold
    thresholded_predictions = predicted_probs - ONE_HALF
    level_predictions = 1.0 / (1.0 + tf.exp(BETA * thresholded_predictions))  # [B, L]
    # predictions = tf.reduce_prod(level_predictions, axis=-1, keepdims=True)  # [B, 1]
    predictions = tf.exp(tf.reduce_sum(tf.log(level_predictions), axis=-1, keepdims=True))  # [B, 1]

    # Compute the (approximate) F1 score
    f1_score = 2 * tf.reduce_sum(predictions * labels) / (tf.reduce_sum(predictions) + tf.reduce_sum(labels))
    return 1.0 - f1_score
