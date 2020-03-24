import numpy as np
from enum import Enum, auto
from collections import namedtuple, defaultdict
from typing import DefaultDict, Dict, Union, List, Optional, Tuple

from utils.constants import SMALL_NUMBER
from utils.np_utils import precision, recall, f1_score


class ClassificationMetric(Enum):
    ACCURACY = auto()
    PRECISION = auto()
    RECALL = auto()
    LATENCY = auto()
    F1_SCORE = auto()
    LEVEL = auto()
    FLOPS = auto()


class RegressionMetric(Enum):
    MSE = auto()
    LATENCY = auto()
    LEVEL = auto()
    FLOPS = auto()


SummaryMetrics = namedtuple('SummaryMetrics', ['mean', 'geom_mean', 'std', 'geom_std', 'median', 'first_quartile', 'third_quartile', 'minimum', 'maximum'])
SaturationMetrics = namedtuple('SaturationMetrics', ['low', 'high'])
Prediction = namedtuple('Prediction', ['sample_id', 'prediction', 'expected'])


HIGH_SATURATION = 0.9
LOW_SATURATION = 0.1


def get_classification_metric(metric_name: ClassificationMetric, model_output: np.ndarray, expected_output: np.ndarray, latency: float, level: int, flops: Union[int, float]) -> float:
    if metric_name == ClassificationMetric.ACCURACY:
        return float(np.average(1.0 - np.abs(model_output - expected_output)))
    elif metric_name == ClassificationMetric.RECALL:
        return float(recall(model_output, expected_output))
    elif metric_name == ClassificationMetric.PRECISION:
        return float(precision(model_output, expected_output))
    elif metric_name == ClassificationMetric.F1_SCORE:
        return float(f1_score(model_output, expected_output))
    elif metric_name == ClassificationMetric.LATENCY:
        return latency
    elif metric_name == ClassificationMetric.LEVEL:
        return float(level)
    elif metric_name == ClassificationMetric.FLOPS:
        return float(flops)
    else:
        raise ValueError(f'Unknown metric name {metric_name}')


def get_regression_metric(metric_name: RegressionMetric, model_output: np.ndarray, expected_output: np.ndarray, latency: float, level: int, flops: Union[int, float]) -> float:
    if metric_name == RegressionMetric.MSE:
        return np.average(np.square(model_output - expected_output))
    elif metric_name == RegressionMetric.LATENCY:
        return latency
    elif metric_name == RegressionMetric.LEVEL:
        return float(level)
    elif metric_name == RegressionMetric.FLOPS:
        return float(flops)
    else:
        raise ValueError(f'Unknown metric name {metric_name}')


def absolute_percentage_error(predictions: Dict[str, List[Prediction]]) -> Dict[str, SummaryMetrics]:
    ape_summaries: Dict[str, SummaryMetrics] = dict()

    for pred_op, predictions in predictions.items():
        errors: List[float] = []
        for pred in predictions:
            abs_perc_error = abs((pred.prediction - pred.expected) / (pred.expected + 1e-8))
            errors.append(abs_perc_error)

        metrics = SummaryMetrics(mean=np.average(errors),
                                 geom_mean=geometric_mean(errors),
                                 std=np.std(errors),
                                 median=np.std(errors),
                                 first_quartile=np.percentile(errors, 25),
                                 third_quartile=np.percentile(errors, 75),
                                 minimum=np.min(errors),
                                 maximum=np.max(errors))
        ape_summaries[pred_op] = metrics
    return ape_summaries


def geometric_mean(array: Union[np.array, List[float]]) -> float:
    log_arr = np.log(np.array(array) + SMALL_NUMBER)
    return np.exp(np.sum(log_arr) / len(log_arr))


def geometric_standard_deviation(array: Union[np.array, List[float]], mean: float) -> float:
    squared_log_ratio = np.square(np.log(np.array(array) / (mean + SMALL_NUMBER) + SMALL_NUMBER))
    return np.sqrt(np.average(squared_log_ratio))


def get_summaries(errors_dict: Union[Dict[str, List[float]], DefaultDict[str, List[float]]]) -> Dict[str, SummaryMetrics]:
    metrics_dict: Dict[str, SummaryMetrics] = dict()
    for series, values in errors_dict.items():
        geom_mean = geometric_mean(values)
        metrics_dict[series] = SummaryMetrics(mean=np.average(values),
                                              geom_mean=geom_mean,
                                              std=np.std(values),
                                              geom_std=geometric_standard_deviation(values, geom_mean),
                                              median=np.std(values),
                                              first_quartile=np.percentile(values, 25),
                                              third_quartile=np.percentile(values, 75),
                                              minimum=np.min(values),
                                              maximum=np.max(values))
    return metrics_dict


def gate_saturation_levels(gate_values: Union[Dict[str, List[float]], DefaultDict[str, List[float]]]) -> Dict[str, Dict[str, SaturationMetrics]]:
    
    saturation_dict: Dict[str, Dict[str, SaturationMetrics]] = defaultdict(dict)
    for series, gate_dict in gate_values.items():

        for gate_name, gate_values in gate_dict.items():
            low, high = 0, 0
            for val in gate_values:
                if val <= LOW_SATURATION:
                    low += 1
                elif val >= HIGH_SATURATION:
                    high += 1
            saturation_dict[series][gate_name] = SaturationMetrics(low=float(low) / len(gate_values),
                                                                   high=float(high) / len(gate_values))

    return saturation_dict


class TestMetrics:

    def __init__(self, squared_error: Union[Dict[str, List[float]], DefaultDict[str, List[float]]],
                 abs_error: Union[Dict[str, List[float]], DefaultDict[str, List[float]]],
                 abs_percentage_error: Union[Dict[str, List[float]], DefaultDict[str, List[float]]],
                 latency: Union[Dict[str, List[float]], DefaultDict[str, List[float]]],
                 gate_values: Union[Dict[str, List[float]], DefaultDict[str, List[float]]],
                 predictions: Union[Dict[str, List[Prediction]], DefaultDict[str, List[Prediction]]]):
        self.squared_error = get_summaries(squared_error)
        self.abs_error = get_summaries(abs_error)
        self.abs_percentage_error = get_summaries(abs_percentage_error)
        self.latency = get_summaries(latency)
        self.gate_saturation = gate_saturation_levels(gate_values)
        self.predictions = predictions

    def __getitem__(self, key: str) -> Optional[Dict[str, SummaryMetrics]]:
        key = key.lower()
        if key == 'squared_error':
            return self.squared_error
        elif key == 'abs_error':
            return self.abs_error
        elif key == 'abs_percentage_error':
            return self.abs_percentage_error
        elif key == 'latency':
            return self.latency
        return None


def __str__(self) -> str:
    return f'Squared Error: {self.squared_error}\n' + \
            f'Absolute Error: {self.abs_error}\n' + \
            f'Absolute Percentage Error: {self.abs_percentage_error}\n' + \
            f'Latency: {self.latency}'
