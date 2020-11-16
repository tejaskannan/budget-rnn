import re
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from scipy import stats
from typing import Dict, DefaultDict, List, Optional, Tuple

from utils.file_utils import read_by_file_suffix, iterate_files
from utils.testing_utils import ClassificationMetric
from utils.constants import SMALL_NUMBER
from plotting.plotting_constants import MARKER_SIZE, STYLE, SMALL_FONT, NORMAL_FONT, LARGE_FONT
from plotting.plotting_utils import get_results, to_label, ModelResult, make_noise_generator, select_adaptive_system
from plotting.plotting_utils import rename_dataset, get_model_name, get_fill, MODEL_ORDER


SummarizedResult = namedtuple('SummarizedResult', ['mean', 'std', 'median', 'first', 'third'])

WIDTH = 0.4
STRIDE = 1.9
Y_MARGIN = 0.01
X_MARGIN = 0.15


def aggregate_results(model_results: DefaultDict[float, Dict[str, List[ModelResult]]]) -> Dict[str, SummarizedResult]:
    normalized_results: DefaultDict[str, List[float]] = defaultdict(list)  # Key: model name, Value: List of normalized accuracy scores

    for budget, results in model_results.items():
        for model_name, system_results in results.items():
            # Accuracy Normalization
            acc = np.average([r.accuracy for r in system_results])
            normalized_results[model_name].append(acc)

    result: Dict[str, SummarizedResult] = dict()
    for model_name, accuracy in normalized_results.items():
        result[model_name] = SummarizedResult(mean=np.average(accuracy),
                                              std=np.std(accuracy),
                                              median=np.median(accuracy),
                                              first=np.percentile(accuracy, 25),
                                              third=np.percentile(accuracy, 75))
    return result


def flatten_datasets(dataset_results: Dict[str, Dict[str, SummarizedResult]],
                     baseline_mode: str,
                     series_mode: str) -> Tuple[DefaultDict[str, List[SummarizedResult]], List[str]]:
    flattened: DefaultDict[str, List[SummarizedResult]] = defaultdict(list)
    datasets: List[str] = []

    fixed_rnn = 'RNN FIXED_{0}'.format(baseline_mode.upper())
    fixed_skip_rnn = 'SKIP_RNN FIXED_{0}'.format(baseline_mode.upper())
    fixed_phased_rnn = 'PHASED_RNN FIXED_{0}'.format(baseline_mode.upper())
    fixed_sample_rnn = 'SAMPLE_RNN FIXED_{0}'.format(baseline_mode.upper())
    adaptive_sample_rnn = 'SAMPLE_RNN ADAPTIVE'
    randomized_sample_rnn = 'SAMPLE_RNN RANDOMIZED'

    for dataset, model_results in sorted(dataset_results.items()):  # By sorting, we keep the lists in a consistent order

        for model_name, aggregate_results in model_results.items():

            should_keep = False
            if series_mode == 'all':
                should_keep = model_name in (fixed_rnn, fixed_skip_rnn, fixed_phased_rnn, fixed_sample_rnn, adaptive_sample_rnn, randomized_sample_rnn)
            elif series_mode == 'baseline':
                should_keep = model_name in (fixed_rnn, fixed_skip_rnn, fixed_phased_rnn, adaptive_sample_rnn)
                model_name = get_model_name(model_name)
            elif series_mode == 'sample':
                should_keep = model_name in (fixed_sample_rnn, randomized_sample_rnn, adaptive_sample_rnn)
            else:
                raise ValueError('Unknown series mode: {0}'.format(series_mode))

            if should_keep:
                flattened[model_name].append(aggregate_results)

        datasets.append(dataset)

    return flattened, datasets


def plot(normalized_results: Dict[str, Dict[str, SummarizedResult]],
         aggregation_type: str,
         output_file: Optional[str],
         shift: float,
         show_errorbars: bool,
         add_annotations: bool,
         baseline_mode: str,
         series_mode: str,
         sensor_type: str):

    flattened, datasets = flatten_datasets(normalized_results,
                                           baseline_mode=baseline_mode,
                                           series_mode=series_mode)
    avg_improvement: Dict[str, SummarizedResult] = dict()

    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots(figsize=(10, 8))

        xs = np.arange(len(datasets) + 1) * STRIDE
        num_series = len(flattened)
        offset = -((num_series - 1) / 2) * WIDTH

        model_idx = 0
        for model_name in MODEL_ORDER:
            if model_name not in flattened:
                continue

            results = flattened[model_name]

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

            avg_improvement[model_name] = SummarizedResult(mean=np.average(aggregate_values),
                                                           std=np.std(aggregate_values),
                                                           first=np.percentile(aggregate_values, 25),
                                                           third=np.percentile(aggregate_values, 75),
                                                           median=np.median(aggregate_values))

            # Include the `Summary` series
            if aggregation_type == 'mean':
                aggregate_values.append(np.average(aggregate_values))
                lower_errors.append(np.std(aggregate_values))
                upper_errors.append(np.std(aggregate_values))
            elif aggregation_type == 'median':
                aggregate_values.append(np.median(aggregate_values))
                lower_errors.append(np.median(aggregate_values) - np.percentile(aggregate_values, 25))
                upper_errors.append(np.percentile(aggregate_values, 75) - np.median(aggregate_values))

            ax.bar(xs + offset, aggregate_values, width=WIDTH, label=to_label(model_name), color=get_fill(model_name), edgecolor='k')

            if show_errorbars:
                ax.errorbar(xs + offset, aggregate_values, xerr=None, yerr=[lower_errors, upper_errors], capsize=2, ecolor='black', ls='none')

            if add_annotations:
                for x, y in zip(xs, aggregate_values):
                    if model_idx == 0:
                        xshift = offset - 3.5 * X_MARGIN
                        if y >= 0.2:
                            xshift -= X_MARGIN
                    elif model_idx == 1:
                        if y > 0.2:
                            xshift = offset - 2 * X_MARGIN
                        elif y >= 0:
                            xshift = offset - 3 * X_MARGIN
                        else:
                            xshift = offset - 0.5 * X_MARGIN
                    else:
                        xshift = offset - 0.5 * X_MARGIN
                        if y >= 0.2:
                            xshift += 0.5 * X_MARGIN

                    if y >= 0:
                        yshift = Y_MARGIN
                    else:
                        yshift = -1.5 * Y_MARGIN

                    ax.annotate(xy=(x + offset, y), s='{0:.2f}'.format(y), xytext=(x + xshift, y + yshift), fontsize=SMALL_FONT)

            offset += WIDTH
            model_idx += 1

        ax.legend(fontsize=SMALL_FONT, loc='upper left')

        # Set ticks to dataset names
        datasets.append('All')
        ax.set_xticks(xs)
        ax.set_xticklabels(datasets, fontsize=NORMAL_FONT)

        # Denote the aggregate series
        ax.axvline((xs[-1] + xs[-2]) / 2, linestyle='--', color='k', linewidth=0.5)

        # Set gridline to denote the x-axis
        ax.axhline(0, linestyle='-', color='k', linewidth=0.5)

        if abs(shift) < SMALL_NUMBER:
            ax.set_title('{0} Accuracy Using a {1} Energy Profile'.format(aggregation_type.capitalize(), sensor_type.capitalize()), fontsize=LARGE_FONT)
        else:
            ax.set_title('{0} Accuracy with Noise Bias {1:.2f} Using a {2} Power Profile'.format(aggregation_type.capitalize(), shift, sensor_type.capitalize()), fontsize=LARGE_FONT)

        ax.set_xlabel('Dataset', fontsize=NORMAL_FONT)
        ax.set_ylabel('Average Accuracy', fontsize=NORMAL_FONT)

        for model_name, summary in sorted(avg_improvement.items()):
            print('{0}: Avg -> {1:.3f} Std -> {2:.3f}'.format(model_name, summary.mean, summary.std))

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
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
    parser.add_argument('--baseline-mode', type=str, choices=['under_budget', 'max_accuracy'], required=True)
    parser.add_argument('--aggregation-type', type=str, choices=['mean', 'median'], required=True)
    parser.add_argument('--show-errorbars', action='store_true')
    parser.add_argument('--add-annotations', action='store_true')
    parser.add_argument('--sensor-type', choices=['bluetooth', 'temperature'], default='temperature')
    parser.add_argument('--series-mode', choices=['all', 'sample', 'baseline'], default='all')
    args = parser.parse_args()

    noise_generator = make_noise_generator(noise_type=args.noise_type,
                                           noise_loc=args.noise_loc,
                                           noise_scale=args.noise_scale,
                                           noise_period=args.noise_period,
                                           noise_amplitude=args.noise_amplitude)

    # Get the test results for adaptive and baseline models. Key is the budget, Value
    # is a dictionary mapping the model name to a list of accuracy values.
    model_results = get_results(input_folders=args.input_folders,
                                noise_generator=noise_generator,
                                model_type='rnn',
                                baseline_mode=args.baseline_mode)

    # Track the name of the dataset
    dataset_name = None

    # Summarize the accuracy values on each data-set
    summarized_results = {rename_dataset(dataset_name): aggregate_results(results) for dataset_name, results in model_results.items()}

    # Plot the results
    plot(summarized_results,
         args.aggregation_type,
         output_file=args.output_file,
         shift=noise_generator.loc,
         show_errorbars=args.show_errorbars,
         add_annotations=args.add_annotations,
         baseline_mode=args.baseline_mode,
         series_mode=args.series_mode,
         sensor_type=args.sensor_type)
