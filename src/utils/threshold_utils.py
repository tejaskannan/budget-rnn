import numpy as np
from collections import namedtuple
from typing import Union, List
from enum import Enum, auto

from .constants import SMALL_NUMBER, BIG_NUMBER


ThresholdedOutput = namedtuple('ThresholdedOutput', ['predictions', 'indices'])
TwoSidedThreshold = namedtuple('TwoSidedThreshold', ['lower', 'upper'])


class InferenceMode(Enum):
    BINARY_ONE_SIDED = auto()
    BINARY_TWO_SIDED = auto()


def order_threshold_lists(lower: List[float], upper: List[float], should_sort: bool) -> List[TwoSidedThreshold]:
    thresholds = [TwoSidedThreshold(*t) for t in zip(lower, upper)]
    return order_thresholds(thresholds, should_sort=should_sort)


def order_thresholds(thresholds: List[TwoSidedThreshold], should_sort: bool) -> List[TwoSidedThreshold]:
    """
    Sorts the given thresholds and returns a copy.
    """
    lower = [t.lower for t in thresholds]
    upper = [t.upper for t in thresholds]

    sorted_thresholds: List[TwoSidedThreshold] = []

    if should_sort:
        lower = sorted(lower)
        upper = sorted(upper)

    for low, high in zip(lower, upper):
        min_val, max_val = min(low, high), max(low, high)
        sorted_thresholds.append(TwoSidedThreshold(lower=low, upper=high))

    return sorted_thresholds


def adaptive_inference(predicted_probs: np.ndarray, thresholds: List[TwoSidedThreshold], mode: InferenceMode) -> ThresholdedOutput:
    if mode == InferenceMode.BINARY_ONE_SIDED:
        return lower_threshold_predictions(predicted_probs, thresholds)
    elif mode == InferenceMode.BINARY_TWO_SIDED:
        return two_sided_predictions(predicted_probs, thresholds)

    raise ValueError(f'Unknown inference mode: {mode}')


def lower_threshold_predictions(predicted_probs: np.ndarray, thresholds: List[TwoSidedThreshold]) -> ThresholdedOutput:
    """
    Adaptive inference algorithm for binary classification based only on lower thresholds.

    Args:
        predicted_probs: A [B, L] matrix of predicted probabilities for each model level.
        thresholds: A list of [L] thresholds. Only the lower thresholds are used.
    Returns:
        A pair of (1) predictions for each sample and (2) the number of levels computed. Both are [B] arrays.
    """
    assert predicted_probs.shape[1] == len(thresholds), 'Must have as many thresholds as outputs.'

    thresholds_arr = np.expand_dims([t.lower for t in thresholds], axis=0)  # [1, L]
    level_outputs = np.greater(predicted_probs, thresholds_arr).astype(dtype=np.float32)  # [B, L]

    unified_outputs = np.prod(level_outputs, axis=-1)  # [B]

    # Compute indices of 'stopped' levels
    num_thresholds = len(thresholds)
    index_range = np.arange(start=0, stop=num_thresholds)
    zero_indices = np.where(level_outputs == 0, index_range, num_thresholds - 1)
    indices = np.min(zero_indices, axis=-1)

    return ThresholdedOutput(predictions=unified_outputs, indices=indices)


def two_sided_predictions(predicted_probs: np.ndarray, thresholds: List[TwoSidedThreshold]) -> ThresholdedOutput:
    """
    Adaptive inference algorithm for binary classification tasks when given two-sided thresholds. This algorithm
    actually computes outputs for all levels and is a simulated version of the step-wise inference algorithm.

    Args:
        predicted_probs: A [B, L] matrix of predicted probabilities for each level of the model
        thresholds: A [L] list of upper and lower threshold pairs
    Returns:
        A pair of (1) predictions for each sample and (2) the number of levels computed for each sample. Both
        elements are [B] arrays.
    """
    assert predicted_probs.shape[1] == len(thresholds), 'Must have as many thresholds as outputs'

    # Unpack the thresholds
    upper_thresholds = np.expand_dims([t.upper for t in thresholds], axis=0)  # [1, L]
    lower_thresholds = np.expand_dims([t.lower for t in thresholds], axis=0)  # [1, L]

    num_thresholds = len(thresholds)
    threshold_indices = np.arange(start=0, stop=num_thresholds)  # [L]

    # Compute outputs and levels based on upper thresholds
    upper_level_outputs = np.greater(predicted_probs, upper_thresholds).astype(dtype=np.float32)  # [B, L]
    upper_outputs = np.any(upper_level_outputs, axis=-1).astype(dtype=np.float32)  # [B]
 
    # A [B, K] matrix that is i when the prob[i] is > the upper threshold and L otherwise
    upper_indices = np.where(upper_level_outputs == 1, threshold_indices, num_thresholds)
    upper_levels = np.min(upper_indices, axis=-1)  # [B]

    # Compute outputs and levels based on lower thresholds
    lower_level_outputs = np.greater(predicted_probs, lower_thresholds).astype(dtype=np.float32)  # [B, L]
    lower_outputs = np.all(lower_level_outputs, axis=-1).astype(dtype=np.float32)  # [B]

    # A [B, K] matrix that is i when a prob[i] is <= the lower threshold and L - 1 otherwise
    lower_indices = np.where(lower_level_outputs == 0, threshold_indices, num_thresholds - 1)
    lower_levels = np.min(lower_indices, axis=-1)  # [B]

    # Unify the threshold outputs
    predictions = np.where(upper_levels < lower_levels, upper_outputs, lower_outputs)  # [B]
    indices = np.minimum(upper_levels, lower_levels)  # [B]

    return ThresholdedOutput(predictions=predictions, indices=indices)
