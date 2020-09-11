import re
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import namedtuple, defaultdict, OrderedDict
from typing import Dict, List, DefaultDict, Tuple, Optional

from utils.file_utils import read_by_file_suffix, iterate_files
from utils.testing_utils import ClassificationMetric
from utils.constants import SMALL_NUMBER


OPT_TEST_LOG_REGEX = re.compile('.*model-optimized-logistic-power-([0-9]+)-test-log-([^-]+)-([^-]+)-.*jsonl\.gz')
TEST_LOG_REGEX = re.compile('.*model-test-log-([^-]+)-([^-]+)-.*jsonl\.gz')

POWER = [24.085, 32.776, 37.897, 43.952, 48.833, 50.489, 54.710, 57.692, 59.212, 59.251]
WIDTH = 0.23

ModelResult = namedtuple('ModelResult', ['power', 'accuracy'])


def get_text_color(bg_color: Tuple[float]) -> str:
    r, g, b, _ = bg_color
    luma = 0.2126 * r * 255 + 0.7152 * g * 255 + 0.0722 * b * 255

    if (luma < 40):
        return 'white'
    else:
        return 'black'


def fixed_policy_result(test_log: Dict[str, Dict[str, float]], budget: float) -> ModelResult:
    result = ModelResult(power=0, accuracy=0)
    for level, (prediction_level, level_results) in enumerate(sorted(test_log.items())):
        if POWER[level] > budget:
            break

        accuracy = level_results[ClassificationMetric.ACCURACY.name]
        if accuracy > result.accuracy:
            result = ModelResult(power=POWER[level], accuracy=accuracy)

    return result


def plot_results(results: DefaultDict[str, Dict[str, ModelResult]], budget: float, output_file: Optional[str]):
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.subplots_adjust(right=0.8)

    color_map = plt.get_cmap('tab20b')
    dataset_xs = np.arange(start=1, stop=len(results) + 1) * 2

    for dataset_idx, (dataset_name, dataset_results) in enumerate(sorted(results.items())):
        x_offset = dataset_xs[dataset_idx] - WIDTH * len(dataset_results) / 2
        color_indices = np.linspace(start=0, stop=1, num=len(dataset_results), endpoint=True)

        for model_idx, (model_name, model_result) in enumerate(sorted(dataset_results.items())):
            color = color_map(color_indices[model_idx])

            ax.bar(x_offset, model_result.accuracy, width=WIDTH, label=model_name, color=color, linewidth=2, edgecolor='black')
            
            # Label the average power consumption
            ax.annotate('{0:.2f}mW'.format(model_result.power),
                        xy=(x_offset, model_result.accuracy),
                        xytext=(x_offset - WIDTH / 3, model_result.accuracy / 2),
                        rotation=90,
                        color=get_text_color(color))

            # Label the accuracy
            ax.annotate('{0:.3f}'.format(model_result.accuracy),
                        xy=(x_offset, model_result.accuracy),
                        xytext=(x_offset - WIDTH / 2.5, model_result.accuracy + 0.01),
                        fontsize=7)

            x_offset += WIDTH

    xtick_labels = list(sorted(results.keys()))
    ax.set_xticks(dataset_xs)
    ax.set_xticklabels(xtick_labels)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy')

    # Remove Duplicates from the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))

    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.25,1), borderaxespad=0)
    ax.set_title('Test Accuracy Results For Average Power Budget of {0:.1f}mW'.format(budget))

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folders', type=str, nargs='*')
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--budget', type=float, required=True)
    args = parser.parse_args()

    results: DefaultDict[str, Dict[str, ModelResult]] = defaultdict(dict)

    for folder in args.input_folders:
        for test_log_path in iterate_files(folder, pattern=r'.*jsonl\.gz'):
           
            # Fetch all optimized results matching the power budget
            opt_match = OPT_TEST_LOG_REGEX.match(test_log_path)
            if opt_match is not None:
                power_budget = int(opt_match.group(1))
                if abs(power_budget - args.budget) < SMALL_NUMBER:
                    model_name = '{0} Optimized'.format(opt_match.group(2).capitalize())
                    dataset = opt_match.group(3).replace('_', ' ')

                    test_log = list(read_by_file_suffix(test_log_path))[0]
                    results[dataset][model_name] = ModelResult(power=test_log['APPROX_POWER'], accuracy=test_log[ClassificationMetric.ACCURACY.name])

            match = TEST_LOG_REGEX.match(test_log_path)
            if match is not None:
                model_name = match.group(1).capitalize()
                dataset = match.group(2).replace('_', ' ')

                test_log = list(read_by_file_suffix(test_log_path))[0]
                if len(test_log) > 1:  # Model has many levels, so we choose the best 'fixed' policy
                    results[dataset][model_name] = fixed_policy_result(test_log, args.budget)
                else:
                    results[dataset][model_name] = ModelResult(power=POWER[-1], accuracy=test_log['model'][ClassificationMetric.ACCURACY.name])

    plot_results(results, budget=args.budget, output_file=args.output_file)
