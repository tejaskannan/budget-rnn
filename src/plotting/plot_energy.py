import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import namedtuple, defaultdict
from typing import List, Dict, Any, DefaultDict, Optional

from utils.file_utils import read_by_file_suffix
from plotting_utils import to_label, rename_dataset, FILL_MAP
from plotting_constants import NORMAL_FONT, SMALL_FONT, LARGE_FONT, STYLE, CAPSIZE


WIDTH = 0.2
STRIDE = 1.2
XMARGIN = 0.1
YMARGIN = 0.03
ComparisonResult = namedtuple('ComparisonResult', ['mean', 'std', 'median', 'first', 'third', 'geom_mean'])
MODEL_NAMES = ['RNN', 'PHASED_RNN', 'SKIP_RNN', 'SAMPLE_RNN']


def geometric_mean(x: np.ndarray) -> float:
    n = len(x)
    return pow(np.prod(x), 1.0 / n)


def merge(comparison_logs: List[Dict[str, Dict[str, Any]]], datasets: List[str]) -> Dict[str, Dict[str, ComparisonResult]]:
    merged = dict()
    for dataset, log in zip(datasets, comparison_logs):
        dataset_name = rename_dataset(dataset)
        merged[dataset_name] = dict()

        for system_name, system_results in log.items():
            comparison = np.array(system_results['raw']) + 1

            comp = ComparisonResult(mean=np.average(comparison),
                                    std=np.std(comparison),
                                    median=np.median(comparison),
                                    first=np.percentile(comparison, 25),
                                    third=np.percentile(comparison, 75),
                                    geom_mean=geometric_mean(comparison))
            model_name = system_name.split()[0]

            merged[dataset_name][model_name] = comp

    return merged


def plot(comparison_logs: Dict[str, Dict[str, ComparisonResult]], output_file: Optional[str]):

    # Invert the logs
    model_results: Dict[str, List[float]] = defaultdict(list)
    datasets: List[str] = []

    for dataset_name, log in sorted(comparison_logs.items()):
        datasets.append(dataset_name)

        for model_name, result in log.items():
            model_results[model_name].append(result)

    with plt.style.context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 9))

        xs = np.arange(len(datasets) + 1) * STRIDE
        offset = -WIDTH * (len(model_results)) / 2

        for idx, model_name in enumerate(MODEL_NAMES):

            avg_energy = [r.geom_mean for r in model_results[model_name]]

            if len(avg_energy) == 0:
                avg_energy = [1.0 for _ in range(len(xs) - 1)]

            avg = geometric_mean(avg_energy)

            ys = avg_energy + [avg]
            label = to_label(model_name)
            color = FILL_MAP[label.replace('_', ' ')]
            ax.bar(xs + offset, ys, linewidth=1, edgecolor='k', width=WIDTH, label=label, color=color)

            # Place the annotations. The shifting is data-specific to prevent overlaps.
            for i, (x, y) in enumerate(zip(xs, ys)):
                yshift = YMARGIN

                if idx == 0:
                    xshift = offset - 3 * XMARGIN
                    if i == 0:
                        yshift = 0.5 * YMARGIN
                    elif i == 4:
                        xshift = offset - 2 * XMARGIN
                    elif i == 7:
                        xshift = offset - 2.5 * XMARGIN
                elif idx == 1:
                    if i == 0:
                        xshift = offset - 2 * XMARGIN
                    elif i == 2:
                        xshift = offset - 3 * XMARGIN
                    else:
                        xshift = offset - XMARGIN
                elif idx == 2:
                    xshift = offset - 0.5 * XMARGIN
                    if i == 3:
                        yshift = 3.5 * YMARGIN
                        xshift = offset - XMARGIN
                else:
                    if i == 3:
                        xshift = offset
                    else:
                        xshift = offset - 0.75 * XMARGIN

                ax.annotate(xy=(x + offset, y), s='{0:1.2f}'.format(y), xytext=(x + xshift, y + yshift), fontsize=SMALL_FONT)

            offset += WIDTH

        # Set x-ticks to the data-set names
        ax.set_xticks(xs)
        ax.set_xticklabels(datasets + ['All'], fontsize=16)

        # increase the y-tick font size
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16)

        # Set gridline to denote the x-axis
        ax.axhline(0, linestyle='-', color='k', linewidth=1)

        # Create a vertical line to denote the `All` category
        ax.axvline((xs[-2] + xs[-1]) / 2, linestyle='--', color='k', linewidth=0.5)

        ax.legend(fontsize=16)
        ax.set_title('Mean Normalized Budget Required for Accuracy Equal to the Budget RNN', fontsize=22)
        ax.set_xlabel('Dataset', fontsize=18)
        ax.set_ylabel('Mean Normalized Energy Budget', fontsize=18)

        plt.tight_layout()
        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file, bbox_type='tight')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folders', type=str, required=True, nargs='+')
    parser.add_argument('--output-file', type=str)
    args = parser.parse_args()

    dataset_names = [t.split('/')[-1] if len(t.split('/')[-1]) > 0 else t.split('/')[-2] for t in args.input_folders]

    comparison_logs = [list(read_by_file_suffix(os.path.join(log, 'energy_comparison.jsonl.gz')))[0] for log in args.input_folders]

    merged = merge(comparison_logs, datasets=dataset_names)
    plot(merged, output_file=args.output_file)
