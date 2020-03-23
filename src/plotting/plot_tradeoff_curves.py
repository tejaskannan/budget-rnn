import matplotlib.pyplot as plt
import re
import os.path
from argparse import ArgumentParser
from typing import Dict, Any, List, Optional

from utils.file_utils import read_by_file_suffix, make_dir


MODEL_TYPE_REGEX = re.compile(r'.*model-test-log-([^-]+)-.*')
MODEL = 'model'
SCHEDULED_MODEL = 'scheduled_model'
LATENCY = 'LATENCY'
MARKER_SIZE = 5


def get_model_type(test_log_file: str) -> str:
    match = MODEL_TYPE_REGEX.match(test_log_file)
    return match.group(1)


def fetch_logs(test_log_files: List[str]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, List[float]] = dict()
    for test_log_file in test_log_files:
        test_log = list(read_by_file_suffix(test_log_file))[0]
        model_type = get_model_type(test_log_file)

        model_results = test_log.get(MODEL, test_log.get(SCHEDULED_MODEL))
        assert model_results is not None, f'Could not retrieve results for {model_type}'

        result[model_type] = model_results

    return result


def plot_tradeoff(model_results: Dict[str, Dict[str, Any]], metric: str, output_folder: Optional[str]):
    with plt.style.context('ggplot'):
        fig, ax = plt.subplots()

        for series, result_dict in sorted(model_results.items()):
            latency = result_dict[LATENCY] * 1000.0
            metric_value = result_dict[metric.upper()]

            ax.plot(metric_value, latency, marker='o', markersize=MARKER_SIZE, label=series)

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

    # Fetch model_results
    model_results = fetch_logs(args.test_logs)

    # Create plots
    for metric in args.metrics:
        plot_tradeoff(model_results, metric, args.output_folder)
