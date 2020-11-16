import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, Tuple, List, DefaultDict, Optional

from plotting.plotting_utils import get_results, to_label, ModelResult, make_noise_generator, select_adaptive_system
from plotting.plotting_utils import rename_dataset, get_model_name, get_fill, MODEL_ORDER


STRIDE = 1
WIDTH = 0.2


def get_budget_diff(model_results: Dict[str, Dict[str, Dict[str, List[ModelResult]]]]) -> Dict[str, DefaultDict[str, float]]:
    perc_diff: Dict[str, DefaultDict[str, List[float]]] = dict()

    for dataset_name, dataset_results in model_results.items():
        perc_diff[dataset_name] = defaultdict(list)

        for budget_str, budget_results in dataset_results.items():
            budget = float(budget_str)

            for model_name, result in budget_results.items():
                power = np.average([r.power for r in result])
                # diff = ((budget - power) / budget) * 100
                utilization = power / budget
                perc_diff[dataset_name][model_name].append(utilization)

    avg_diff: Dict[str, Dict[str, float]] = dict()

    for dataset_name, dataset_diff in perc_diff.items():
        avg_diff[dataset_name] = dict()

        for model_name, model_diff in dataset_diff.items():
            avg_diff[dataset_name][model_name] = (np.average(model_diff), np.std(model_diff))

    return avg_diff


def plot(perc_diff: Dict[str, DefaultDict[str, List[float]]], model_names: List[str], series_mode: str, sensor_type: str, output_file: Optional[str]):

    with plt.style.context('seaborn-ticks'):
        fig, ax = plt.subplots(figsize=(12, 9))

        dataset_names = [rename_dataset(name) for name in sorted(perc_diff.keys())]
        xs = np.arange(len(dataset_names) + 1) * STRIDE
        offset = (-(len(model_names) - 1) / 2) * WIDTH

        for model_name in model_names:
            values = [perc_diff[dataset_name][model_name] for dataset_name in sorted(perc_diff.keys())]
            diff = list(map(lambda t: t[0], values))
            err = list(map(lambda t: t[1], values))

            avg = np.average(diff)
            std = np.std(diff)

            if series_mode == 'baseline':
                model_name = get_model_name(model_name)

            ys = diff + [avg]

            print('Model Name: {0}, Avg Utiliazation: {1}'.format(model_name, ys))

            ax.bar(xs + offset, ys, width=WIDTH, label=to_label(model_name), linewidth=1, edgecolor='k', color=get_fill(model_name))
            # ax.errorbar(xs + offset, diff + [avg], yerr=err + [std], color='k', capsize=2, xerr=None, ls='none')

            offset += WIDTH

        ax.legend()
        ax.set_xlabel('Dataset')
        ax.set_ylabel('% Less than Budget')
        ax.set_title('Average % Energy Consumed Less than Budget on {0} Profile'.format(sensor_type.capitalize()))

        ax.set_xticks(xs)
        ax.set_xticklabels(dataset_names + ['All'])

        ax.axvline((xs[-2] + xs[-1]) / 2, color='k', linestyle='--', linewidth=1)
        ax.axhline(0, color='k', linewidth=1)

        if output_file is not None:
            plt.savefig(output_file)
        else:
            plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folders', type=str, nargs='+', required=True)
    parser.add_argument('--noise-loc', type=float, default=0.0)
    parser.add_argument('--noise-scale', type=float, default=0.01)
    parser.add_argument('--baseline-mode', type=str, choices=['under_budget', 'max_accuracy'], required=True)
    parser.add_argument('--series-mode', choices=['all', 'sample', 'baseline'], default='baseline')
    parser.add_argument('--sensor-type', choices=['bluetooth', 'temperature'], default='temperature')
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    noise_generator = make_noise_generator(noise_type='gaussian',
                                           noise_loc=args.noise_loc,
                                           noise_scale=args.noise_scale,
                                           noise_period=None,
                                           noise_amplitude=None)

    # Fetch the model results
    model_results = get_results(input_folders=args.input_folders,
                                noise_generator=noise_generator,
                                model_type='rnn',
                                baseline_mode=args.baseline_mode)

    perc_diff = get_budget_diff(model_results)

    model_names = ['RNN FIXED_UNDER_BUDGET', 'SKIP_RNN FIXED_UNDER_BUDGET', 'PHASED_RNN FIXED_UNDER_BUDGET', 'SAMPLE_RNN ADAPTIVE']

    plot(perc_diff, model_names, series_mode=args.series_mode, sensor_type=args.sensor_type, output_file=args.output_file)
