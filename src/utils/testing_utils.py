import numpy as np
from enum import Enum, auto
from collections import namedtuple, defaultdict
from typing import DefaultDict, Dict, Union, List, Optional, Tuple
from sklearn.metrics import f1_score, recall_score, precision_score

from utils.constants import SMALL_NUMBER


class ClassificationMetric(Enum):
    ACCURACY = auto()
    PRECISION = auto()
    RECALL = auto()
    MICRO_F1_SCORE = auto()
    MACRO_F1_SCORE = auto()


class RegressionMetric(Enum):
    MSE = auto()
    MAE = auto()
    MAPE = auto()
    LATENCY = auto()
    LEVEL = auto()
    FLOPS = auto()


SummaryMetrics = namedtuple('SummaryMetrics', ['mean', 'geom_mean', 'std', 'geom_std', 'median', 'first_quartile', 'third_quartile', 'minimum', 'maximum'])
SaturationMetrics = namedtuple('SaturationMetrics', ['low', 'high'])
Prediction = namedtuple('Prediction', ['sample_id', 'prediction', 'expected'])


ALL_LATENCY = 'ALL_LATENCY'
HIGH_SATURATION = 0.9
LOW_SATURATION = 0.1


def get_binary_classification_metric(metric_name: ClassificationMetric, model_output: np.ndarray, expected_output: np.ndarray) -> float:
    if metric_name == ClassificationMetric.ACCURACY:
        return float(np.average((model_output == expected_output).astype(float)))
    elif metric_name == ClassificationMetric.RECALL:
        return float(recall_score(expected_output, model_output, average='binary'))
    elif metric_name == ClassificationMetric.PRECISION:
        return float(precision_score(expected_output, model_output, average='binary'))
    elif metric_name in (ClassificationMetric.MICRO_F1_SCORE, ClassificationMetric.MACRO_F1_SCORE):
        return float(f1_score(expected_output, model_output, average='binary'))
    else:
        raise ValueError(f'Unknown metric name {metric_name}')


def get_multi_classification_metric(metric_name: ClassificationMetric,
                                    model_output: np.ndarray,
                                    expected_output: np.ndarray,
                                    num_classes: int) -> float:
    if metric_name == ClassificationMetric.ACCURACY:
        return float(np.average((model_output == expected_output).astype(float)))
    elif metric_name == ClassificationMetric.RECALL:
        return float(recall_score(expected_output, model_output, average='macro'))
    elif metric_name == ClassificationMetric.PRECISION:
        return float(precision_score(expected_output, model_output, average='macro'))
    elif metric_name == ClassificationMetric.MICRO_F1_SCORE:
        return float(f1_score(expected_output, model_output, average='micro'))
    elif metric_name == ClassificationMetric.MACRO_F1_SCORE:
        return float(f1_score(expected_output, model_output, average='macro'))
    else:
        raise ValueError(f'Unknown metric name {metric_name}')


def get_regression_metric(metric_name: RegressionMetric, model_output: np.ndarray, expected_output: np.ndarray, latency: float, level: int, flops: Union[int, float]) -> float:
    if metric_name == RegressionMetric.MSE:
        return np.average(np.square(model_output - expected_output))
    elif metric_name == RegressionMetric.MAE:
        return np.average(np.abs(model_output - expected_output))
    elif metric_name == RegressionMetric.MAPE:
        return np.average(np.abs((model_output - expected_output) / expected_output))
    elif metric_name == RegressionMetric.LATENCY:
        return latency
    elif metric_name == RegressionMetric.LEVEL:
        return float(level)
    elif metric_name == RegressionMetric.FLOPS:
        return float(flops)
    else:
        raise ValueError(f'Unknown metric name {metric_name}')
