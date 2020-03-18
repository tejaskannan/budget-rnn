import numpy as np
from typing import List, Any, Optional, Dict


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
