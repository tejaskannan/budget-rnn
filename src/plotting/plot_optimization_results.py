import os
import re
import numpy as np
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from typing import Dict, Any, List, Optional

from utils.file_utils import read_by_file_suffix
from utils.constants import OPTIMIZED_RESULTS
from utils.testing_utils import ClassificationMetric
from plotting.plotting_constants import STYLE, LABEL_REGEX, LABEL_FORMAT


WIDTH = 0.35


def get_label(test_log_path: str) -> str:
    match = LABEL_REGEX.match(test_log_path)
    name, level_weight = match.group(1), match.group(2)
    return LABEL_FORMAT.format(name, level_weight)


def plot_opt_results(test_logs: List[List[Dict[str, Any]]], labels: List[str], model_name: str, output_file: Optional[str]):
    with plt.style.context(STYLE):

        metrics = [ClassificationMetric.MICRO_F1_SCORE.name, ClassificationMetric.ACCURACY.name, ClassificationMetric.LEVEL.name]
        fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(9, 6))

        for ax, metric in zip(axes, metrics):

            x = 0
            for label, test_log in zip(labels, test_logs):
                metric_values = [log[metric] for log in test_log]
                metric_avg = np.average(metric_values)
                metric_std = np.std(metric_values)

                ax.bar(x, height=metric_avg, width=WIDTH, yerr=metric_std, capsize=2, label=label.capitalize())

                x += 0.5

            metric_label = metric.replace('_', ' ').capitalize()
            ax.set_ylabel(metric_label)

            ax.set_xticks([])

        axes[0].legend(prop={ 'size': 7 })  # Only set legend for the first plot (all other plots have the same series)

        fig.suptitle(f'Average Metric Values for Post-Processing Optimization for the {model_name.capitalize()} Model')
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-logs', type=str, nargs='+')
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    labels: List[str] = []
    test_logs: List[List[Dict[str, Any]]] = []
    for test_log_file in sorted(args.test_logs):
        test_log = list(read_by_file_suffix(test_log_file))[0]
        test_logs.append(test_log[OPTIMIZED_RESULTS])

        label = get_label(test_log_file)
        labels.append(label)

    plot_opt_results(test_logs, labels, args.model_name, args.output_file)
