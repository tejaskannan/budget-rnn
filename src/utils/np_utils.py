import numpy as np
from typing import List, Union, Any, Dict, Optional, Tuple
from collections import namedtuple

from utils.constants import SMALL_NUMBER, BIG_NUMBER


def pad_array(arr: np.array, new_size: int, value: Any, axis: int) -> np.array:
    pad_width = new_size - arr.shape[axis]
    if pad_width <= 0 :
        return arr

    widths = [(0, 0) for _ in range(len(arr.shape))]
    widths[axis] = (0, pad_width)
    return np.pad(arr, widths, mode='constant', constant_values=value)


def softmax(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    exp_array = np.exp(arr)
    return exp_array / np.sum(exp_array, axis=axis, keepdims=True)


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


def multiclass_precision(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the precision for each class

    Args:
        predictions: A [B] array of predictions in the range [0, K)
        labels: A [B]  array of labels in the range [0, K]
        num_classes: The number of classes (K)
    Returns:
        A tuple of two elements:
            (1) An array of [K] elements where each element is the precision for the corresponding class
            (2) A binary array of [K] elements which is 0 when there are no samples for this class
            
    """
    assert predictions.shape == labels.shape, f'Predictions array must have the same shape ({predictions.shape}) as the labels array ({labels.shape})'
    assert np.max(predictions) < num_classes and np.min(predictions) >= 0, f'Predictions must be in the range [0, num_classes)'
    assert np.max(labels) < num_classes and np.min(labels) >= 0, f'Labels must be in the range [0, num_classes)'

    precisions: List[float] = []
    mask: List[int] = []
    for class_id in range(num_classes):
        predictions_eq = np.equal(predictions, class_id)

        true_positives = np.sum(np.logical_and(predictions_eq, np.equal(labels, class_id)).astype(float))  # [B]
        false_positives = np.sum(np.logical_and(predictions_eq, np.not_equal(labels, class_id)).astype(float)) # [B]
        total = true_positives + false_positives

        if total < SMALL_NUMBER:
            precisions.append(0.0)
            mask.append(0)
        else:
            precisions.append(true_positives / total)
            mask.append(1)

    return np.array(precisions), np.array(mask)


def multiclass_recall(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the precision for each class

    Args:
        predictions: A [B] array of predictions in the range [0, K)
        labels: A [B]  array of labels in the range [0, K]
        num_classes: The number of classes (K)
    Returns:
        A tuple of two elements:
            (1) An array of [K] elements where each element is the recall for the corresponding class
            (2) A binary array of [K] elements which is 0 when there are no samples for this class
            
    """
    assert predictions.shape == labels.shape, f'Predictions array must have the same shape ({predictions.shape}) as the labels array ({labels.shape})'
    assert np.max(predictions) < num_classes and np.min(predictions) >= 0, f'Predictions must be in the range [0, num_classes)'
    assert np.max(labels) < num_classes and np.min(labels) >= 0, f'Labels must be in the range [0, num_classes)'

    recalls: List[float] = []
    mask: List[int] = []
    for class_id in range(num_classes):
        labels_eq = np.equal(labels, class_id)

        true_positives = np.sum(np.logical_and(np.equal(predictions, class_id), labels_eq).astype(float))  # [B]
        false_negatives = np.sum(np.logical_and(np.not_equal(predictions, class_id), labels_eq).astype(float)) # [B]
        total = true_positives + false_negatives

        if total < SMALL_NUMBER:
            recalls.append(0.0)
            mask.append(0)
        else:
            recalls.append(true_positives / total)
            mask.append(1)

    return np.array(recalls), np.array(mask)


def multiclass_f1_score(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """
    Multiclass F1 score using the average of the F1 scores of individual classes.

    Args:
        predictions: A [B] array of predictions in the range [0, K)
        labels: A [B] array of labels in the range [0, K)
        num_classes: The number of classes (K)
    Returns:
        The multiclass F1 score.
    """
    precisions, precision_mask = multiclass_precision(predictions, labels, num_classes)
    recalls, recall_mask = multiclass_recall(predictions, labels, num_classes)

    mask = np.logical_or(precision_mask, recall_mask).astype(float)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + SMALL_NUMBER)
    f1_score = np.sum(f1_scores * mask) / (np.sum(mask) + SMALL_NUMBER)
    return float(f1_score)


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
