import numpy as np
from typing import List, Union, Any, Dict, Optional
from collections import namedtuple

from .constants import SMALL_NUMBER, BIG_NUMBER


ThresholdedOutput = namedtuple('ThresholdedOutput', ['predictions', 'indices'])


def pad_array(arr: np.array, new_size: int, value: Any, axis: int) -> np.array:
    pad_width = new_size - arr.shape[axis]
    if pad_width <= 0 :
        return arr

    widths = [(0, 0) for _ in range(len(arr.shape))]
    widths[axis] = (0, pad_width)
    return np.pad(arr, widths, mode='constant', constant_values=value)


def softmax(arr: np.ndarray) -> np.ndarray:
    exp_array = np.exp(arr)
    return exp_array / np.sum(exp_array)


def sigmoid(arr: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-arr))


def linear_normalize(arr: np.ndarray) -> np.ndarray:
    arr_sum = np.sum(arr)
    arr_sum = arr_sum if abs(arr_sum) > SMALL_NUMBER else SMALL_NUMBER
    return arr / arr_sum


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    l2_norm = np.linalg.norm(arr, ord=2)
    return arr / (l2_norm + SMALL_NUMBER)


def clip_by_norm(arr: np.ndarray, clip: float) -> np.ndarray:
    assert clip > 0, 'The clip value must be positive'
    norm = np.linalg.norm(arr, ord=2)

    # Don't clip arrays that already satisfy the constraint
    if norm < clip:
        return arr

    factor = clip / norm
    return arr * factor


def precision(predictions: np.ndarray, labels: np.ndarray) -> float:
    assert predictions.shape == labels.shape, f'Predictions array must have same shape {predictions.shape} as the labels array {labels.shape}.'

    true_positives = np.sum(predictions * labels)
    false_positives = np.sum(predictions * (1.0 - labels))
    return float(true_positives / (true_positives + false_positives + SMALL_NUMBER))


def recall(predictions: np.ndarray, labels: np.ndarray) -> float:
    assert predictions.shape == labels.shape, f'Predictions array must have same shape {predictions.shape} as the labels array {labels.shape}.'

    true_positives = np.sum(predictions * labels)
    false_negatives = np.sum((1.0 - predictions) * labels)
    return float(true_positives / (true_positives + false_negatives + SMALL_NUMBER))


def f1_score(predictions: np.ndarray, labels: np.ndarray) -> float:
    p = precision(predictions, labels)
    r = recall(predictions, labels)

    return 2 * (p * r) / (p + r + SMALL_NUMBER)


def thresholded_predictions(predicted_probs: np.ndarray, thresholds: Union[List[float], np.ndarray]) -> ThresholdedOutput:
    assert predicted_probs.shape[1] == len(thresholds), 'Must have as many thresholds as outputs.'

    thresholds_arr = np.expand_dims(thresholds, axis=0)  # [1, L]
    level_outputs = np.greater(predicted_probs, thresholds_arr).astype(dtype=np.float32)  # [B, L]

    unified_outputs = np.prod(level_outputs, axis=-1)  # [B]

    # Compute indices of 'stopped' levels
    index_range = np.arange(start=0, stop=len(thresholds))
    zero_indices = np.where(level_outputs == 0, index_range, len(thresholds) - 1)
    indices = np.min(zero_indices, axis=-1)

    return ThresholdedOutput(predictions=unified_outputs, indices=indices)
