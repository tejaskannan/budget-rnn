import re
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import namedtuple, defaultdict, OrderedDict
from typing import Dict, List, DefaultDict, Tuple, Optional, Any

from utils.file_utils import read_by_file_suffix, iterate_files
from utils.testing_utils import ClassificationMetric
from utils.constants import SMALL_NUMBER
from controllers.pid_control import interpolate_power, get_accuracy_index
from controllers.logistic_regression_controller import get_power_for_levels, POWER


OPT_TEST_LOG_REGEX = re.compile('.*model-optimized-logistic-power-([0-9\.]+)-test-log-([^-]+)-([^-]+)-.*jsonl\.gz')
TEST_LOG_REGEX = re.compile('.*model-test-log-([^-]+)-([^-]+)-.*jsonl\.gz')

WIDTH = 0.2

ModelResult = namedtuple('ModelResult', ['power', 'accuracy'])
BaselineResults = namedtuple('BaselineResults', ['budget', 'accuracy', 'name'])


def get_text_color(bg_color: Tuple[float]) -> str:
    r, g, b, _ = bg_color
    luma = 0.2126 * r * 255 + 0.7152 * g * 255 + 0.0722 * b * 255

    if (luma < 40):
        return 'white'
    else:
        return 'black'


def budget_policy_result(test_log: Dict[str, Dict[str, float]], budget: float, power: np.ndarray) -> ModelResult:
    result = ModelResult(power=0, accuracy=0)

    power_estimates = interpolate_power(power, len(test_log))

    for level in range(len(test_log)):
        if power_estimates[level] > budget:
            break

        key = 'prediction_{0}'.format(level)
        level_results = test_log[key]

        accuracy = level_results[ClassificationMetric.ACCURACY.name]
        if accuracy > result.accuracy:
            result = ModelResult(power=power_estimates[level], accuracy=accuracy)

    return result


def accuracy_policy_result(test_log: Dict[str, Dict[str, float]], target_accuracy: float, power: np.ndarray) -> ModelResult:
    result = ModelResult(power=0, accuracy=0)
    power_estimates = interpolate_power(power, len(test_log))

    accuracies: List[float] = []
    for level in range(len(test_log)):
        key = 'prediction_{0}'.format(level)
        level_results = test_log[key]

        level_accuracy = level_results[ClassificationMetric.ACCURACY.name]
        accuracies.append(level_accuracy)

    accuracy_index = get_accuracy_index(target_accuracy, np.array(accuracies))

    return ModelResult(power=power_estimates[accuracy_index], accuracy=accuracies[accuracy_index])


def add_model_result(model_result: ModelResult, model_name: str, x_pos: float, ax: Any, color: Tuple[float], pos: int):
    ax.bar(x_pos, model_result.accuracy, width=WIDTH, label=model_name, linewidth=2, edgecolor='black', color=color)
    
    # Label the average power consumption
    ax.annotate('{0:.2f}mW'.format(model_result.power),
                xy=(x_pos, model_result.accuracy),
                xytext=(x_pos - WIDTH / 3, model_result.accuracy / 2),
                rotation=90,
                color=get_text_color(color))

    # Label the accuracy
    x_div = 2.5 if pos == 2 else 1.25
    y_shift = 0.03 if pos == 1 else 0.01
    ax.annotate('{0:.3f}'.format(model_result.accuracy),
                xy=(x_pos, model_result.accuracy),
                xytext=(x_pos - WIDTH / x_div, model_result.accuracy + y_shift),
                fontsize=7)


def plot_results(adaptive_results: DefaultDict[float, Dict[str, ModelResult]],
                 baseline_results: DefaultDict[float, Dict[str, BaselineResults]],
                 budgets: List[float],
                 dataset: str,
                 output_file: Optional[str]):
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.subplots_adjust(right=0.8)

    color_map = plt.get_cmap('Spectral')
    xs = np.arange(start=1, stop=len(budgets) + 1) * 3

    # Get all model - policy combinations for consistent coloring
    series_names: Set[str] = set()
    for baseline_result in baseline_results.values():
        for adaptive_model_name, baseline in baseline_result.items():
            series_names.add(adaptive_model_name.upper())
            series_names.add('{0} BUDGET'.format(baseline.name.upper()))
            series_names.add('{0} ACCURACY'.format(baseline.name.upper()))

    series_names = list(sorted(series_names))
    color_ids = np.linspace(start=0.0, stop=1.0, num=len(series_names), endpoint=True)

    for budget_idx, budget in enumerate(sorted(budgets)):
        adaptive_model_results = adaptive_results[budget]
        baseline_model_results = baseline_results[budget]

        num_bars = len(adaptive_model_results) * 4  # Two baselines for each adaptive model plus space

        x_offset = xs[budget_idx] - WIDTH * (num_bars / 2)

        for model_idx, (model_name, model_result) in enumerate(sorted(adaptive_model_results.items())):
            # Plot the adaptive model
            color_idx = series_names.index(model_name.upper())
            add_model_result(model_result, model_name.upper(), x_offset, ax, color=color_map(color_ids[color_idx]), pos=0)
            x_offset += WIDTH

            # Plot the budget baseline policy and accuracy baseline policy
            baseline = baseline_model_results[model_name]
            acc_name = '{0} ACCURACY'.format(baseline.name.upper())
            color_idx = series_names.index(acc_name)
            add_model_result(baseline.accuracy, model_name=acc_name, x_pos=x_offset, ax=ax, color=color_map(color_ids[color_idx]), pos=1)
            x_offset += WIDTH

            budget_name = '{0} BUDGET'.format(baseline.name.upper())
            color_idx = series_names.index(budget_name)
            add_model_result(baseline.budget, model_name=budget_name, x_pos=x_offset, ax=ax, color=color_map(color_ids[color_idx]), pos=2)
            x_offset += 2 * WIDTH


    xtick_labels = list(sorted(adaptive_results.keys()))
    ax.set_xticks(xs)
    ax.set_xticklabels(xtick_labels)

    ax.set_xlabel('Budget')
    ax.set_ylabel('Accuracy')

    # Remove Duplicates from the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))

    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.25,1), borderaxespad=0)
    ax.set_title('Test Accuracy Results On {0}'.format(dataset.upper()))

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--budgets', type=float, nargs='+')
    parser.add_argument('--num-levels', type=int, required=True)
    args = parser.parse_args()

    adaptive_results: DefaultDict[float, Dict[str, ModelResult]] = defaultdict(dict)
    baseline_results: DefaultDict[float, Dict[str, BaselineResults]] = defaultdict(dict)
    num_levels = args.num_levels
    folder = args.input_folder
    dataset = None

    for budget in args.budgets:
        dataset_results: Dict[str, Dict[str, ModelResult]] = dict()

        power_estimates = get_power_for_levels(POWER, num_levels)

        # Fetch the adaptive model results
        for test_log_path in iterate_files(folder, pattern=r'.*jsonl\.gz'):
            # Fetch all optimized results matching the power budget
            opt_match = OPT_TEST_LOG_REGEX.match(test_log_path)
            if opt_match is not None:
                power_budget = float(opt_match.group(1))
                if abs(power_budget - budget) < SMALL_NUMBER:
                    model_name = opt_match.group(2).capitalize().replace('_', ' ')
                    dataset = opt_match.group(3).replace('_', ' ')

                    test_log = list(read_by_file_suffix(test_log_path))[0]
                    adaptive_results[budget][model_name] = ModelResult(power=test_log['APPROX_POWER'], accuracy=test_log[ClassificationMetric.ACCURACY.name])

        for test_log_path in iterate_files(folder, pattern=r'.*jsonl\.gz'):
            match = TEST_LOG_REGEX.match(test_log_path)
            if match is not None:
                model_name = match.group(1).capitalize()
                dataset = match.group(2).replace('_', ' ')

                # Get the baseline results for each adaptive model
                for adaptive_model_name, model_result in adaptive_results[budget].items():
                    if model_name.lower() == 'nbow' and adaptive_model_name.lower() != 'adaptive nbow':
                        continue
                    if model_name.lower() == 'rnn' and adaptive_model_name.lower() not in ('sample', 'cascade', 'bidir_sample'):
                        continue

                    test_log = list(read_by_file_suffix(test_log_path))[0]
                    budget_result = budget_policy_result(test_log, budget, power_estimates)

                    accuracy_result = accuracy_policy_result(test_log, model_result.accuracy, power_estimates)

                    baseline_results[budget][adaptive_model_name] = BaselineResults(budget=budget_result, accuracy=accuracy_result, name=model_name)

    plot_results(adaptive_results=adaptive_results,
                 baseline_results=baseline_results,
                 budgets=args.budgets,
                 dataset=dataset,
                 output_file=args.output_file)
