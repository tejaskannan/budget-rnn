import numpy as np
from typing import Tuple

from utils.np_utils import min_max_normalize, round_to_precision
from utils.constants import BIG_NUMBER, SMALL_NUMBER


def level_entropy(labels: np.ndarray, levels: np.ndarray, num_classes: int, num_levels: int) -> np.ndarray:
    """
    Args:
        labels: A [B, 1] array of labels
        levels: A [S, B] array of computed levels
        num_classes: The number of classes (C)
    Returns:
        A [S] array containing the entropy scores.
    """
    entropy = np.zeros(shape=(levels.shape[0], ))
    for class_idx in range(num_classes):
        label_mask = np.squeeze((labels == class_idx).astype(float))  # [B]
        label_fraction = np.average(label_mask)

        for i in range(levels.shape[0]):
            level_counts = np.bincount(levels[i, :], weights=label_mask, minlength=num_levels)  # [B]
            level_distribution = level_counts / (np.sum(level_counts) + SMALL_NUMBER)

            entropy[i] += label_fraction * np.sum(-1 * level_distribution * np.log(level_distribution + SMALL_NUMBER))

    return entropy


def tanh_approx(x: np.ndarray) -> np.ndarray:
    """
    Approximation of tanh using polynomials. We use this equation during inference with fixed point operations.
    """
    return np.clip(x * (0.5 + np.square(0.25 * x)) / (0.5 + np.square(0.5 * x)), a_min=-1, a_max=1)


def normalize_logits(logits: np.ndarray, precision: int) -> np.ndarray:
    """
    Normalizes the logits for thresholded predictions

    Args:
        logits: A [B, L, C] array of normalized log probabilities
        precision: Fixed point precision to evaluate under
    Returns:
        A [B, L, C] normalized logits array
    """
    if logits.shape[-1] > 2:
        normalized_logits = min_max_normalize(logits, axis=-1)
    else:
        # When the number of classes is 2, we use the sigmoid function to approximate
        # the behavior of softmax.
        sigmoid_diff = 0.5 * (tanh_approx(0.5 * (logits[:, :, 0] - logits[:, :, 1])) + 1)  # [B, L]
        sigmoid_diff = np.expand_dims(sigmoid_diff, axis=-1)  # [B, L, 1]

        normalized_logits = np.concatenate([sigmoid_diff, 1 - sigmoid_diff], axis=-1)  # [B, L, 2]

    return round_to_precision(normalized_logits, precision)



def threshold_predictions(predictions: np.ndarray, thresholds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the predictions using the early-stopping inference algorithm.

    Args:
        predictions: [B, L, C] array of normalized log probabilities for each sample, level, and class
        thresholds: [L] or [S, L] array of thresholds for each level
    Returns:
        A tuple of (1) A [S, B] array with the predictions per sample and (2) A [S, B] array with the number of computed levels.
        If S = 1, then we squeeze the arrays for convenience and return [B] arrays.
    """
    # Reshape thresholds to a [1, L] array if needed. We refer to the first dimension as (S)
    if len(thresholds.shape) == 1:
        thresholds = np.expand_dims(thresholds, axis=0)

    # [S, 1, L]
    expanded_thresholds = np.expand_dims(thresholds, axis=1)

    # Create mask using the maximum probability
    max_prob = np.expand_dims(np.max(predictions, axis=-1), axis=0)  # [1, B, L]
    diff_mask = (max_prob < expanded_thresholds).astype(np.float32) * BIG_NUMBER  # [S, B, L]

    indices = np.arange(start=0, stop=thresholds.shape[-1]).reshape((1, 1, -1))  # [1, 1, L]

    # Apply mask to compute the number of computed levels
    masked_indices = indices + diff_mask  # [S, B, L]
    levels = np.clip(np.min(masked_indices, axis=-1).astype(int), a_min=0, a_max=predictions.shape[1] - 1)  # [S, B]

    # Use number of levels to get the classification
    predicted_class_per_level = np.argmax(predictions, axis=-1)  # [B, L]
    batch_indices = np.arange(start=0, stop=predictions.shape[0])  # [B]

    if levels.shape[0] == 1:
        levels = np.squeeze(levels, axis=0)
        predicted_classes = predicted_class_per_level[batch_indices, levels]  # [B]
    else:
        predicted_classes = np.vstack([predicted_class_per_level[batch_indices, levels[i, :]] for i in range(levels.shape[0])])  # [S, B]

    return predicted_classes, levels


def optimal_levels(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Finds the first level which predicts the output correctly. If no level, exists
    the optimal level is 0 (to minimize computation).

    Args:
        logits: A [B, L, C] array of log probabilities for each sample (B), level (L), and class (C)
        labels: A [B] array of labels
    """
    max_levels = logits.shape[1]

    # Compute the level-wise predictions
    predicted_classes = np.argmax(logits, axis=-1)  # [B, L]
    expanded_labels = np.reshape(labels, newshape=(-1, 1))  # [B, 1]
    correct_predictions = (predicted_classes == expanded_labels).astype(float)  # [B, L]

    # Create the indices mask
    indices = np.expand_dims(np.arange(start=0, stop=max_levels), axis=0)  # [B, 1]
    predictions_mask = (1.0 - correct_predictions) * BIG_NUMBER
    masked_indices = predictions_mask + indices  # [B, L]

    min_levels = np.min(masked_indices, axis=-1)  # [B]
    optimal_levels = np.where(min_levels > max_levels, 0, min_levels)
    return optimal_levels
