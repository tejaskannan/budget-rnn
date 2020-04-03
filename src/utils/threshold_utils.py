import numpy as np
from collections import namedtuple
from typing import Union, List
from enum import Enum, auto

from utils.constants import SMALL_NUMBER, BIG_NUMBER


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


def matrix_to_thresholds(threshold_matrix: np.ndarray) -> List[TwoSidedThreshold]:
    """
    Converts a [L, 2] matrix of thresholds into a list of two-sided threshold tuples.
    """
    thresholds: List[TwoSidedThreshold] = []
    for entry in threshold_matrix:
        thresholds.append(TwoSidedThreshold(lower=np.min(entry), upper=np.max(entry)))
    return thresholds


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


def upper_threshold_predictions(predicted_probs: np.ndarray, thresholds: np.ndarray) -> ThresholdedOutput:
    """
    Adaptive Inference algorithm using upper-bounded thresholds. This algorithm stops execution when an observed
    probability is above its corresponding threshold.

    Args:
        predicted_probs: A [B, L, K] matrix of predicted class probabilities (L) for each level (K) in the batch (B)
        thresholds: A [L, K] matrix of thresholds for each level (L) and class (K)
    Returns:
        A pair of (1) predictions for each sample and (2) the number of levels computed for each sample. Both elements
        are [B] arrays.
    """
    assert predicted_probs.shape[1:] == thresholds.shape, 'Misaligned shapes for predictions and thresholds'

    num_thresholds, num_classes = thresholds.shape
    threshold_indices = np.arange(start=0, stop=num_thresholds)
    class_indices = np.arange(start=0, stop=num_classes)

    # Compute the predicted index for each level
    batch_indices = np.expand_dims(np.arange(start=0, stop=predicted_probs.shape[0]), axis=-1) # [B, 1]
    batch_indices = batch_indices * num_thresholds * num_classes  # [B, 1]
    print(batch_indices)

    threshold_indices = np.expand_dims(threshold_indices * num_classes, axis=0)
    index_shift = np.repeat(batch_indices, repeats=num_thresholds, axis=-1) + threshold_indices  # [B, L]
    
    print(index_shift)
    predicted_indices = np.argmax(predicted_probs, axis=-1)  # [B, L]
    shifted_predictions = index_shift + predicted_indices
    print(shifted_predictions)

    max_probs = np.take(predicted_probs, indices=shifted_predictions)  # [B, L]
    selected_thresholds = np.take(thresholds, indices=shifted_predictions)  # [B, L]

    print(predicted_probs)
    print(predicted_indices)
    print(max_probs)
    # print(selected_thresholds)

    # Find elements in which the computation can be truncated
    threshold_indices = np.arange(start=0, stop=thresholds.shape[0])
    threshold_comparison = np.greater(max_probs, selected_thresholds).astype(dtype=np.float32)  # [B, L]
    threshold_indices = np.where(threshold_comparison == 1, threshold_indices, thresholds.shape[0])  # [B, L]
    computed_levels = np.min(threshold_indices, axis=-1)

    # print(computed_levels)



predicted_probs = np.array([[[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]], [[0.3, 0.4, 0.3], [0.35, 0.2, 0.45]]])
thresholds = np.array([[0.7, 0.6, 0.4], [0.8, 0.7, 0.6]])

upper_threshold_predictions(predicted_probs, thresholds)
