import numpy as np
import math
from typing import Union, List, Dict

from tracking_utils.constants import SMALL_NUMBER


def normalize(array: np.array):
    norm = np.linalg.norm(array, ord=2, axis=1)
    return array / np.expand_dims(norm, axis=1)


def softmax(array: Union[np.array, List[float]]) -> np.array:
    array_exp = np.exp(array)
    exp_sum = np.sum(array_exp)
    return array_exp / exp_sum

def get_random_point(points: List[Dict[str, List[float]]]) -> List[float]:
    bound_index = np.random.randint(low=0, high=len(points))
    bound = points[bound_index]

    low, high = bound['low'], bound['high']

    point = []
    for low_val, high_val in zip(low, high):
        if abs(high_val - low_val) < SMALL_NUMBER:
            point.append(low_val)
        else:
            p = np.random.uniform(low=low_val, high=high_val + SMALL_NUMBER)
            point.append(p)

    return point
