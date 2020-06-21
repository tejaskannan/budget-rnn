import numpy as np
from typing import Tuple

from utils.constants import BIG_NUMBER


def threshold_predictions(predictions: np.ndarray, thresholds: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the predictions using the early-stopping inference algorithm.

    Args:
        predictions: [B, L, C] array of normalized log probabilities for each sample, level, and class
        thresholds: [L] array of thresholds for each level
    Returns:
        A tuple of (1) A [B] array with the predictions per sample and (2) A [B] array with the number of computed levels.
    """
    # Reshape thresholds to a [1, L] array
    expanded_thresholds = np.expand_dims(thresholds, axis=0)

    # Create mask using the maximum probability
    max_prob = np.max(predictions, axis=-1)  # [B, L]
    diff_mask = (max_prob < expanded_thresholds).astype(np.float32) * BIG_NUMBER  # [B, L]

    indices = np.expand_dims(np.arange(start=0, stop=len(thresholds)), axis=0)  # [1, L]

    # Apply mask to compute the number of computed levels
    masked_indices = indices + diff_mask  # [B, L]
    levels = np.clip(np.min(masked_indices, axis=-1).astype(int), a_min=0, a_max=predictions.shape[1] - 1)  # [B]

    # Use number of levels to get the classification
    predicted_class_per_level = np.argmax(predictions, axis=-1)  # [B, L]
    batch_indices = np.arange(start=0, stop=predictions.shape[0])  # [B]
    predicted_classes = predicted_class_per_level[batch_indices, levels]  # [B]

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
