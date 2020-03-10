import numpy as np
from typing import List, Union
from collections import namedtuple

from .constants import SMALL_NUMBER, BIG_NUMBER


ThresholdedOutput = namedtuple('ThresholdedOutput', ['predictions', 'indices'])


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
