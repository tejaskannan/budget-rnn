import numpy as np
from collections import namedtuple
from typing import DefaultDict, Dict, Union, List, Optional, Tuple


SummaryMetrics = namedtuple('SummaryMetrics', ['mean', 'std', 'median', 'first_quartile', 'third_quartile', 'minimum', 'maximum'])


def get_summaries(errors_dict: Union[Dict[str, List[float]], DefaultDict[str, List[float]]]) -> Dict[str, SummaryMetrics]:
    metrics_dict: Dict[str, SummaryMetrics] = dict()
    for series, values in errors_dict.items():
        metrics_dict[series] = SummaryMetrics(mean=np.average(values),
                                              std=np.std(values),
                                              median=np.std(values),
                                              first_quartile=np.percentile(values, 25),
                                              third_quartile=np.percentile(values, 75),
                                              minimum=np.min(values),
                                              maximum=np.max(values))
    return metrics_dict


class TestMetrics:

    def __init__(self, squared_error: Union[Dict[str, List[float]], DefaultDict[str, List[float]]],
                 abs_error: Union[Dict[str, List[float]], DefaultDict[str, List[float]]],
                 abs_percentage_error: Union[Dict[str, List[float]], DefaultDict[str, List[float]]],
                 latency: Union[Dict[str, List[float]], DefaultDict[str, List[float]]]):
        self.squared_error = get_summaries(squared_error)
        self.abs_error = get_summaries(abs_error)
        self.abs_percentage_error = get_summaries(abs_percentage_error)
        self.latency = get_summaries(latency)

    def __getitem__(self, key: str) -> Optional[SummaryMetrics]:
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
