import os.path
import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from argparse import ArgumentParser
from matplotlib import cm
from typing import Tuple, List, Optional, Dict

from plotting_constants import LINEWIDTH, STYLE, MARKER_SIZE
from utils.file_utils import make_dir


def read_data(energy_path: str, threshold: int) -> List[Tuple[List[float], List[float], float]]:
    with open(energy_path, 'r') as energy_file:
        reader = csv.reader(energy_file, quotechar='"', delimiter=',')

        file_iterator = iter(reader)
        next(file_iterator)  # Skip the file headers

        time: List[float] = []
        power: List[float] = []
        energy: List[float] = []

        for i, line in enumerate(file_iterator):
            # Unpack the arguments in the format below
            # [0] Time (ns)  [1] Current (nA)  [2] Voltage (mV)  [3] Energy (uJ) 
            t = int(line[0]) * 1e-9
            current = int(line[1]) * 1e-9
            voltage = int(line[2]) * 1e-3
            e = float(line[3]) * 1e-6

            # Compute the instantaneous power in milli Watts using the product of current and voltage
            p = current * voltage * 1e3
                
            time.append(t)
            power.append(p)
            energy.append(e)

    # Prune the graph to adjust for start and end times
    start, end = None, len(time)
    window_size = 100
    for index in range(window_size, len(power), window_size):
        data_window = power[index - window_size:index]
        min_val = np.min(data_window)

        if start is None and min_val > threshold:
            start = index

#        if end is None and start is not None and min_val < threshold:
#            end = index - window_size
#            break

    if start is None:
        start = 0

#    if end is None:
#        end = len(time)

    start_energy = energy[start]
    end_energy = energy[end - 1]

    return time[start:end], power[start:end], end_energy - start_energy


def moving_average(x: List[float], window: int) -> List[float]:
    avg: List[float] = []
    for i in range(0, len(x) - window + 1, window):
        data_window = x[i:i+window]
        avg.append(sum(data_window) / len(data_window))

    return avg


def distribution_matrix(power_dict: Dict[str, List[float]]) -> List[List[float]]:
    results: List[List[float]] = []
    for key_one, vals_one in sorted(power_dict.items()):
        test_results: List[float] = []
        for key_two, vals_two in sorted(power_dict.items()):
            t_stat, p_value = stats.ttest_ind(vals_one, vals_two)
            test_results.append(p_value)

        results.append(test_results)

    return results


def plot(energy_paths: List[str], labels: List[str], output_folder: Optional[str], window: Optional[int]):
    if output_folder is not None:
        make_dir(output_folder)

    power_dict: Dict[str, List[float]] = dict()
    energy_dict: Dict[str, float] = dict()

    # Plots the power graphs
    with plt.style.context(STYLE):
        fig, axes = plt.subplots(nrows=len(energy_paths), ncols=1, figsize=(12, 8))

        if not isinstance(axes, tuple):
            axes = (axes, )

        for ax, label, energy_path in zip(axes, labels, energy_paths):
            time, power, total_energy = read_data(energy_path, threshold=50)

            power_dict[label] = power
            energy_dict[label] = total_energy

            title = 'Device Power During Sensing and Inference for {0}'.format(label)

            if window is not None:
                time = moving_average(time, window)
                power = moving_average(power, window)

                title = '{0} Sample Moving Average of {1}'.format(window, title)

            time = np.array(time) - np.min(time)

            ax.plot(time, power, linewidth=1)

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Power (mW)')
            ax.set_title(title)

        plt.tight_layout()

        if output_folder is None:
            plt.show()
        else:
            plt.savefig(os.path.join(output_folder, 'power.pdf'))

        # Computes and plots the matrix of p-values between the power distributions
        p_value_mat = distribution_matrix(power_dict)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.matshow(p_value_mat, cmap=plt.get_cmap('magma_r'))

        for (i, j), z in np.ndenumerate(p_value_mat):
            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

        im = ax.set_title('P-Values from Two-Sided T-Test on Power Distributions')

        if output_folder is None:
            plt.show()
        else:
            plt.savefig(os.path.join(output_folder, 'pvalues.pdf'))

    # Create a latex-style table of total energy and average power
    table_lines: List[str] = ['\\begin{tabular}{ccc}', '\\textbf{Levels}, \\textbf{Total Energy (J)}, \\textbf{Avg Power (mW)} \\\\', '\midrule']
    for level, key in enumerate(sorted(power_dict.keys())):
        line = '{0} & {1:.3f} & {2:.3f} \\\\'.format(level + 1, energy_dict[key], np.average(power_dict[key]))
        table_lines.append(line)

    table_lines.append('\\end{tabular}')
    table_str = '\n'.join(table_lines)

    if output_folder is None:
        print(table_str)
    else:
        with open(os.path.join(output_folder, 'power_table.tex'), 'w') as fout:
            fout.write(table_str)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--energy-files', type=str, nargs='+')
    parser.add_argument('--labels', type=str, nargs='+')
    parser.add_argument('--window', type=int)
    parser.add_argument('--output-folder', type=str)
    args = parser.parse_args()

    assert len(args.energy_files) == len(args.labels), 'Must have as many labels as files.'

    plot(energy_paths=args.energy_files,
         labels=args.labels,
         output_folder=args.output_folder,
         window=args.window)
