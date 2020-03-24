import matplotlib.pyplot as plt
import re
import os.path
from argparse import ArgumentParser
from typing import Dict, Any, List, Optional, Tuple, DefaultDict
from random import random
from collections import defaultdict, namedtuple

from utils.file_utils import read_by_file_suffix, make_dir
from utils.constants import MODEL, SCHEDULED_MODEL, SCHEDULED_GENETIC
from utils.testing_utils import ClassificationMetric
from plotting_constants import STYLE, MARKER_SIZE


MODEL_TYPE_REGEX = re.compile(r'.*model-.*test-log-([^-]+)-.*')
LATENCY_FACTOR = 1000.0

plt.rc({'font.size': 8})

MetricPair = namedtuple('MetricPair', ['latency', 'metric'])


def get_model_type(test_log_file: str) -> str:
    match = MODEL_TYPE_REGEX.match(test_log_file)
    return match.group(1)


def get_metric_value(model_result: Dict[str, Any], metric: str) -> MetricPair:
    latency = model_result[ClassificationMetric.LATENCY.name] * LATENCY_FACTOR
    metric_value = model_result[metric.upper()]
    return MetricPair(latency=latency, metric=metric_value)


def fetch_logs(test_log_files: List[str], metric: str) -> DefaultDict[str, List[MetricPair]]:
    result: DefaultDict[str, List[Tuple[float, float]]] = defaultdict(list)

    for test_log_file in test_log_files:
        test_log = list(read_by_file_suffix(test_log_file))[0]
        model_type = get_model_type(test_log_file).capitalize()

        if MODEL in test_log:
            result[model_type].append(get_metric_value(test_log[MODEL], metric))

        if SCHEDULED_MODEL in test_log:
            result[f'{model_type} Scheduled'].append(get_metric_value(test_log[SCHEDULED_MODEL], metric))

        if SCHEDULED_GENETIC in test_log:
            result[f'{model_type} Optimized'].append(get_metric_value(test_log[SCHEDULED_GENETIC], metric))

        # Compile all other predictions into one series
        for series in sorted(filter(lambda k: k.startswith('prediction'), test_log.keys())):
            result[model_type].append(get_metric_value(test_log[series], metric))

    return result


def plot_tradeoff(model_results: Dict[str, List[MetricPair]], metric: str, output_folder: Optional[str]):
    with plt.style.context(STYLE):
        fig, ax = plt.subplots()

        for series, result_list in sorted(model_results.items()):
            latencies = [pair.latency for pair in result_list]
            metric_values = [pair.metric for pair in result_list]

            ax.plot(metric_values, latencies, marker='o', markersize=MARKER_SIZE, label=series)
            
            # Include data labels
            #for metric_value, latency in zip(metric_values, latencies):
            #    ax.annotate(f'{metric_value:.4f}', (metric_value, latency), (metric_value - X_OFFSET, latency + Y_OFFSET))

        # Formatting for the Metric Name
        metric_tokens = [t.capitalize() for t in metric.split('_')]
        metric_label = ' '.join(metric_tokens)

        ax.set_title('Tradeoff Curve for {0} vs Latency'.format(metric_label))
        ax.set_xlabel(metric_label)
        ax.set_ylabel('Latency (ms)')
        ax.legend()

        if output_folder is None:
            plt.show()
        else:
            make_dir(output_folder)
            output_file = os.path.join(output_folder, f'{metric}.pdf')
            plt.savefig(output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-logs', type=str, nargs='+')
    parser.add_argument('--metrics', type=str, nargs='+')
    parser.add_argument('--output-folder', type=str)
    args = parser.parse_args()

    # Create plots
    for metric in args.metrics:
        model_results = fetch_logs(args.test_logs, metric)
        plot_tradeoff(model_results, metric, args.output_folder)
