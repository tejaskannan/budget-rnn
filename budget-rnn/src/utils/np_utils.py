import numpy as np
from typing import List, Union, Any, Dict, Optional, Tuple
from collections import namedtuple

from utils.constants import SMALL_NUMBER, BIG_NUMBER


def index_of(arr: np.ndarray, value: Union[int, float]) -> int:
    assert len(arr.shape) == 1, 'Array must be 1 dimensional'

    idx = -1
    for i in range(len(arr)):
        if abs(arr[i] - value) < SMALL_NUMBER:
            idx = i
            break

    return idx


def pad_array(arr: np.array, new_size: int, value: Any, axis: int) -> np.array:
    pad_width = new_size - arr.shape[axis]
    if pad_width <= 0:
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


def min_max_normalize(arr: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    min_value = np.min(arr, axis=axis, keepdims=True)
    adjusted_arr = arr - min_value
    arr_sum = np.sum(adjusted_arr, axis=-1, keepdims=True)
    return adjusted_arr / arr_sum


def l2_normalize(arr: np.ndarray) -> np.ndarray:
    l2_norm = np.linalg.norm(arr, ord=2)
    return arr / (l2_norm + SMALL_NUMBER)


def round_to_precision(arr: np.ndarray, precision: int):
    rounded = np.floor(arr * (2 << precision))
    return rounded / (2 << precision)


def clip_by_norm(arr: np.ndarray, clip: float, axis: Optional[int] = None) -> np.ndarray:
    assert clip > 0, 'The clip value must be positive'

    norm = np.linalg.norm(arr, ord=2, axis=axis, keepdims=True)

    n_dims = len(arr.shape)
    reps = [1 for _ in range(n_dims - 1)] + [arr.shape[-1]]
    tiled_norm = np.tile(norm, reps=reps)

    # Don't clip elements that already satisfy the constraint
    clipped_arr = arr * clip / (norm + SMALL_NUMBER)
    return np.where(tiled_norm < clip, arr, clipped_arr)


def np_majority(logits: np.ndarray) -> np.ndarray:
    """
    Returns the majority label across all sequence elements.

    Args:
        logits: A [B, T, D] array of class logits (D) for each sequence element (T) in the batch (B)
    Returns:
        A [B] array of predictions per sample
    """
    predicted_probs = softmax(logits, axis=-1)  # [B, T, D]
    predicted_labels = np.argmax(predicted_probs, axis=-1)  # [B, T]

    predictions: List[int] = []
    for sample_labels in predicted_labels:
        label_counts = np.bincount(sample_labels)
        pred = np.argmax(label_counts)
        predictions.append(pred)

    return np.array(predictions)
