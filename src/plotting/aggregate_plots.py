import re
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from scipy import stats
from typing import Dict, DefaultDict, List, Optional, Tuple

from utils.file_utils import read_by_file_suffix, iterate_files
from utils.testing_utils import ClassificationMetric
from plotting.plotting_constants import MARKER_SIZE
from plotting.plotting_utils import get_results, to_label, ModelResult, make_noise_generator, select_adaptive_system, rename_system
from plotting.plotting_utils import rename_dataset


NormalizedResult = namedtuple('NormalizedResult', ['mean', 'std', 'median', 'first', 'third'])


WIDTH = 0.35
STRIDE = 2
Y_MARGIN = 0.01
X_MARGIN = 0.15


def aggregate_and_normalize(model_results: DefaultDict[float, Dict[str, List[ModelResult]]], baseline_name: str) -> Dict[str, NormalizedResult]:
    normalized_results: DefaultDict[str, List[float]] = defaultdict(list)  # Key: model name, Value: List of normalized accuracy scores

    # Collect the range of the baseline results
    baseline_results: List[float] = []
    for budget, results in model_results.items():
        if baseline_name not in results:
            continue

        baseline_results.append(np.average([r.accuracy for r in results[baseline_name]]))

    min_val, max_val = np.min(baseline_results), np.max(baseline_results)

    for budget, results in model_results.items():
        if baseline_name not in results:
            continue

        baseline_accuracy = np.average([r.accuracy for r in results[baseline_name]])

        for model_name, system_results in results.items():
            if model_name == baseline_name:
                continue

            # Min-Max Normalization
            acc = np.average([r.accuracy for r in system_results])
            normalized_accuracy = (acc - baseline_accuracy) / (max_val - min_val)
            normalized_results[model_name].append(normalized_accuracy)

    result: Dict[str, NormalizedResult] = dict()
    for model_name, normalized_accuracy in normalized_results.items():
        result[model_name] = NormalizedResult(mean=np.average(normalized_accuracy),
                                              std=np.std(normalized_accuracy),
                                              median=np.median(normalized_accuracy),
                                              first=np.percentile(normalized_accuracy, 25),
                                              third=np.percentile(normalized_accuracy, 75))
    return result


def flatten_datasets(dataset_results: Dict[str, Dict[str, NormalizedResult]], to_keep: Dict[str, str], baseline_mode: str) -> Tuple[DefaultDict[str, List[NormalizedResult]], List[str]]:
    flattened: DefaultDict[str, List[NormalizedResult]] = defaultdict(list)
    datasets: List[str] = []

    fixed_rnn = 'RNN FIXED_{0}'.format(baseline_mode.upper())
    fixed_skip_rnn = 'SKIP_RNN FIXED_{0}'.format(baseline_mode.upper())
    fixed_phased_rnn = 'PHASED_RNN FIXED_{0}'.format(baseline_mode.upper())
    fixed_sample_rnn = 'SAMPLE_RNN FIXED_{0}'.format(baseline_mode.upper())
    adaptive_sample_rnn = 'SAMPLE_RNN ADAPTIVE'
    randomized_sample_rnn = 'SAMPLE_RNN RANDOMIZED'

    for dataset, model_results in sorted(dataset_results.items()):  # By sorting, we keep the lists in a consistent order

        for model_name, normalized_results in model_results.items():
            if model_name in (fixed_rnn, fixed_skip_rnn, fixed_phased_rnn, fixed_sample_rnn, adaptive_sample_rnn, randomized_sample_rnn):
                renamed_model = rename_system(model_name)
                flattened[renamed_model].append(normalized_results)

        datasets.append(rename_dataset(dataset))

    return flattened, datasets


def plot(normalized_results: Dict[str, Dict[str, NormalizedResult]],
         aggregation_type: str,
         baseline_name: str,
         output_file: Optional[str],
         shift: float,
         show_errorbars: bool,
         add_annotations: bool,
         best_adaptive_models: Dict[str, str],
         baseline_mode: str):

    flattened, datasets = flatten_datasets(normalized_results, to_keep=best_adaptive_models, baseline_mode=baseline_mode)
    avg_improvement: Dict[str, float] = dict()

    with plt.style.context('ggplot'):
        fig, ax = plt.subplots(figsize=(10, 8))

        xs = np.arange(len(datasets)) * STRIDE
        num_series = len(flattened)
        offset = -((num_series - 1) / 2) * WIDTH

        for model_name, results in sorted(flattened.items()):

            if 'greedy' in model_name.lower():
                continue

            aggregate_values: List[float] = []
            lower_errors: List[float] = []
            upper_errors: List[float] = []
            
            for result in results:
                if aggregation_type == 'mean':
                    aggregate_values.append(result.mean)
                    lower_errors.append(result.std)
                    upper_errors.append(result.std)
                elif aggregation_type == 'median':
                    aggregate_values.append(result.median)
                    lower_errors.append(result.median - result.first)
                    upper_errors.append(result.third - result.median)

            ax.bar(xs + offset, aggregate_values, width=WIDTH, label=to_label(model_name))

            avg_improvement[model_name] = (np.average(aggregate_values), np.min(aggregate_values), np.max(aggregate_values))

            if show_errorbars:
                ax.errorbar(xs + offset, aggregate_values, xerr=None, yerr=[lower_errors, upper_errors], capsize=2, ecolor='black', ls='none')

            if add_annotations:
                for x, y in zip(xs, aggregate_values):
                    sign = 2 * float(y >= 0) - 1
                    ax.annotate(xy=(x + offset, y), s='{0:.2f}'.format(y), xytext=(x + offset - X_MARGIN, y + sign * Y_MARGIN), fontsize=7)

            offset += WIDTH

        ax.legend(fontsize=8)

        # Set ticks to dataset names
        ax.set_xticks(xs)
        ax.set_xticklabels(datasets)

        ax.set_title('{0} Accuracy Improvement vs. the {1} Model with Power Shift {2:.2f}'.format(aggregation_type.capitalize(), to_label(baseline_name), shift))
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Accuracy Improvement')

        for model_name, (avg, min_val, max_val) in sorted(avg_improvement.items()):
            print('{0}: {1:.3f} [{2:.3f}, {3:.3f}]'.format(model_name, avg, min_val, max_val))

        if output_file:
            plt.savefig(output_file)
        else:
            plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folders', type=str, nargs='+', required=True)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--noise-loc', type=float, required=True)
    parser.add_argument('--noise-scale', type=float, required=True)
    parser.add_argument('--noise-type', type=str, required=True)
    parser.add_argument('--noise-period', type=int)
    parser.add_argument('--noise-amplitude', type=int)
    parser.add_argument('--model-type', type=str, choices=['rnn', 'nbow'], required=True)
    parser.add_argument('--baseline-mode', type=str, choices=['under_budget', 'max_accuracy'], required=True)
    parser.add_argument('--aggregation-type', type=str, choices=['mean', 'median'], required=True)
    parser.add_argument('--show-errorbars', action='store_true')
    parser.add_argument('--add-annotations', action='store_true')
    args = parser.parse_args()

    model_type = args.model_type.upper()

    noise_generator = make_noise_generator(noise_type=args.noise_type,
                                           noise_loc=args.noise_loc,
                                           noise_scale=args.noise_scale,
                                           noise_period=args.noise_period,
                                           noise_amplitude=args.noise_amplitude)

    # Get the test results for adaptive and baseline models. Key is the budget, Value
    # is a dictionary mapping the model name to a list of accuracy values.
    model_results = get_results(input_folders=args.input_folders,
                                noise_generator=noise_generator,
                                model_type=args.model_type,
                                baseline_mode=args.baseline_mode)

    baseline_name = '{0} FIXED_{1}'.format(model_type, args.baseline_mode).upper()

    # Track the name of the dataset
    dataset_name = None

    # Summarize the results over all budgets
    for dataset_name, results in model_results.items():
        aggregate_and_normalize(results, baseline_name)

    normalized_results = {dataset_name: aggregate_and_normalize(results, baseline_name) for dataset_name, results in model_results.items()}

    # Plot the results
    plot(normalized_results,
         args.aggregation_type,
         baseline_name=baseline_name,
         output_file=args.output_file,
         shift=noise_generator.loc,
         show_errorbars=args.show_errorbars,
         add_annotations=args.add_annotations,
         best_adaptive_models=dict(),
         baseline_mode=args.baseline_mode)
