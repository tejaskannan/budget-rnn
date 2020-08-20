import re
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, DefaultDict, List, Optional

from utils.file_utils import read_by_file_suffix, iterate_files
from utils.testing_utils import ClassificationMetric
from plotting.plotting_constants import MARKER_SIZE
from plotting.plotting_utils import get_results, ModelResult, to_label, make_noise_generator


def plot_curves(model_results: Dict[str, DefaultDict[float, List[ModelResult]]],
                shift: float,
                dataset_name: str,
                output_file: Optional[str]):

    with plt.style.context('fast'):

        fig, ax = plt.subplots(figsize=(8, 6))

        xs: List[float] = []
        ys: DefaultDict[str, List[float]] = defaultdict(list)  # Map from model name to list of accuracy values for each budget

        for budget, results in sorted(model_results.items()):

            if budget >= 4:
                continue

            for model_name, budget_results in sorted(results.items()):
                accuracy_values = [r.accuracy for r in budget_results]
                avg_accuracy = np.average(accuracy_values)
                ys[model_name].append(avg_accuracy)

            xs.append(budget)
        
        for model_name, accuracy in sorted(ys.items()):
            ax.plot(xs, accuracy, marker='o', markersize=MARKER_SIZE, label=to_label(model_name))

        ax.legend()
        ax.set_xlabel('Budget (mw)')
        ax.set_ylabel('Accuracy')
        ax.set_title('System Accuracy on {0} with Power Shift {1:.2f}'.format(to_label(dataset_name), shift))

        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--noise-loc', type=float, required=True)
    parser.add_argument('--noise-scale', type=float, required=True)
    parser.add_argument('--noise-type', type=str, required=True)
    parser.add_argument('--noise-period', type=int)
    parser.add_argument('--noise-amplitude', type=int)
    parser.add_argument('--model-type', choices=['rnn', 'nbow'], required=True)
    args = parser.parse_args()

    noise_generator = make_noise_generator(noise_type=args.noise_type,
                                           noise_loc=args.noise_loc,
                                           noise_scale=args.noise_scale,
                                           noise_period=args.noise_period,
                                           noise_amplitude=args.noise_amplitude)

    model_type = args.model_type.upper()
    model_results = get_results([args.input_folder], noise_generator)

    dataset_name = list(model_results.keys())[0]

   # Plot the collected results
    plot_curves(model_results=model_results[dataset_name],
                shift=noise_generator._loc,
                dataset_name=dataset_name,
                output_file=args.output_file)
