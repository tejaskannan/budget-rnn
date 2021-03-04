import re
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, DefaultDict, List, Optional, Set

from dataset.dataset import DataSeries
from dataset.dataset_factory import get_dataset
from utils.file_utils import read_by_file_suffix, iterate_files
from utils.testing_utils import ClassificationMetric
from utils.constants import SMALL_NUMBER
from plotting.plotting_constants import MARKER_SIZE, STYLE, NORMAL_FONT, LARGE_FONT
from plotting.plotting_utils import get_results, ModelResult, to_label, make_noise_generator, FILL_MAP


MARKERS = ['o']
MODEL_ORDER = ['RNN Fixed Under Budget', 'Phased RNN Fixed Under Budget', 'Skip RNN Fixed Under Budget', 'Budget RNN Adaptive']


def plot_curves(model_results: Dict[str, DefaultDict[float, List[ModelResult]]],
                shift: float,
                dataset_name: str,
                output_file: Optional[str],
                dataset_size: int,
                sample_rate: float,
                seq_length: float):

    with plt.style.context(STYLE):

        fig, ax = plt.subplots(figsize=(8, 6))

        budgets: Set[float] = set()
        ys: DefaultDict[str, List[float]] = defaultdict(list)  # Map from model name to list of accuracy values for each budget

        for budget, results in sorted(model_results.items()):

            for model_name, budget_results in sorted(results.items()):
                accuracy_values = [r.accuracy for r in budget_results]
                assert len(accuracy_values) == 1, 'Only supports 1 trial per model'

                ys[to_label(model_name)].append(accuracy_values[0])

            budgets.add(round(budget, 3))

        xs = list(sorted(budgets))
        xs = np.array(xs)
        xs = xs * (sample_rate * seq_length * dataset_size)

        for model_name in MODEL_ORDER:

            if model_name.startswith('RNN'):
                system_name = model_name.split()[0]
            else:
                system_name = ' '.join(model_name.split()[0:2])

            accuracy = ys[model_name]

            ax.plot(xs, accuracy,
                    marker='o',
                    markersize=8,
                    label=system_name,
                    linewidth=3,
                    color=FILL_MAP[system_name])

            print('{0}: {1}'.format(system_name, np.average(accuracy)))

        ax.legend(fontsize=LARGE_FONT)
        ax.set_xlabel('Budget (mJ)', fontsize=LARGE_FONT)
        ax.set_ylabel('Accuracy', fontsize=LARGE_FONT)

        ax.tick_params(labelsize=NORMAL_FONT)

        if abs(shift) > SMALL_NUMBER:
            ax.set_title('System Accuracy on the {0} Dataset with a Bias of {1:.2f}'.format(to_label(dataset_name), shift), fontsize=LARGE_FONT)
        else:
            ax.set_title('System Accuracy on the {0} Dataset'.format(to_label(dataset_name)), fontsize=LARGE_FONT)

        plt.tight_layout()
        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file, bbox_inches='tight')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--noise-loc', type=float, required=True)
    parser.add_argument('--noise-scale', type=float, required=True)
    parser.add_argument('--noise-type', type=str, required=True)
    parser.add_argument('--noise-period', type=int)
    parser.add_argument('--noise-amplitude', type=int)
    parser.add_argument('--seq-length', type=int, required=True)
    parser.add_argument('--sample-rate', type=float, default=2)
    parser.add_argument('--dataset-folder', type=str, required=True)
    parser.add_argument('--baseline-mode', type=str, required=True, choices=['under_budget', 'max_accuracy'])
    parser.add_argument('--model-type', choices=['rnn', 'nbow'], required=True)
    args = parser.parse_args()

    noise_generator = make_noise_generator(noise_type=args.noise_type,
                                           noise_loc=args.noise_loc,
                                           noise_scale=args.noise_scale,
                                           noise_period=args.noise_period,
                                           noise_amplitude=args.noise_amplitude)

    model_type = args.model_type.upper()
    model_results = get_results([args.input_folder], noise_generator, model_type=args.model_type, baseline_mode=args.baseline_mode)

    dataset_name = list(model_results.keys())[0]

    dataset = get_dataset(dataset_type='standard', data_folder=args.dataset_folder)
    dataset.dataset[DataSeries.TEST].load()
    test_size = dataset.dataset[DataSeries.TEST].length

    # Plot the collected results
    plot_curves(model_results=model_results[dataset_name],
                shift=noise_generator._loc,
                dataset_name=dataset_name,
                output_file=args.output_file,
                dataset_size=test_size,
                sample_rate=args.sample_rate,
                seq_length=args.seq_length)
