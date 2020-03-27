import numpy as np
from typing import List, Dict, Callable, Any
from collections import namedtuple


ErrorMetrics = namedtuple('ErrorMetrics', ['name', 'mean', 'std', 'minimum', 'maximum'])


def rmse(system_evaluation: List[Dict[str, Any]]) -> ErrorMetrics:
    return calculate_prediction_error(system_evaluation=system_evaluation,
                                      sample_error=lambda r, a: np.sum(np.square(r - a)),
                                      aggregation=lambda e: np.sqrt(np.average(e)),
                                      name='rmse')


def geometric_mean_distance(system_evaluation: List[Dict[str, Any]]) -> ErrorMetrics:
    n_samples = len(system_evaluation)
    return calculate_prediction_error(system_evaluation=system_evaluation,
                                      sample_error=lambda r, a: np.sum(np.square(r - a)),
                                      aggregation=lambda e: np.power(np.prod(e), 1.0 / float(n_samples)),
                                      name='geometric-mean-distance')


def calculate_prediction_error(system_evaluation: List[Dict[str, Any]],
                               sample_error: Callable[[np.ndarray, np.ndarray], float],
                               aggregation: Callable[[List[float]], float],
                               name: str) -> ErrorMetrics:
    errors: List[float] = []
    for sample in system_evaluation:
        prediction, actual = sample['prediction'], sample['actual']
        errors.append(sample_error(prediction, actual))
    
    metrics = ErrorMetrics(name=name,
                           mean=aggregation(errors),
                           std=np.std(errors),
                           minimum=np.min(errors),
                           maximum=np.max(errors))
    return metrics


def arithmetic_mean_sensors(system_evaluation: List[Dict[str, Any]]) -> ErrorMetrics:
    return calculate_communication(system_evaluation=system_evaluation,
                                   aggregation=lambda e: np.average(e),
                                   name='arithmetic-mean')


def geometric_mean_sensors(system_evaluation: List[Dict[str, Any]]) -> ErrorMetrics:
    n_samples = len(system_evaluation)
    return calculate_communication(system_evaluation=system_evaluation,
                                   aggregation=lambda e: np.power(np.prod(e), 1 / float(n_samples)),
                                   name='geometric_mean')


def calculate_communication(system_evaluation: List[Dict[str, Any]],
                            aggregation: Callable[[List[float]], float],
                            name: str) -> ErrorMetrics:
    sample_values: List[float] = []
    for sample in system_evaluation:
        num_sensors = sample['num_sensors']
        sample_values.append(float(num_sensors))

    metrics = ErrorMetrics(name=name,
                           mean=aggregation(sample_values),
                           std=np.std(sample_values),
                           minimum=np.min(sample_values),
                           maximum=np.max(sample_values))
    return metrics
