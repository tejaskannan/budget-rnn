import numpy as np
from argparse import ArgumentParser
from collections import Counter, defaultdict
from typing import Dict, List

from plotting_utils import make_noise_generator, ModelResult, get_results, to_label, rename_dataset


def make_table(results: Dict[str, Dict[float, Dict[str, List[ModelResult]]]], baseline_mode: str):

    skip_rnn = 'SKIP_RNN FIXED_{0}'.format(baseline_mode.upper())
    rnn = 'RNN FIXED_{0}'.format(baseline_mode.upper())
    phased_rnn = 'PHASED_RNN FIXED_{0}'.format(baseline_mode.upper())
    sample_rnn = 'SAMPLE_RNN FIXED_{0}'.format(baseline_mode.upper())
    randomized_sample = 'SAMPLE_RNN RANDOMIZED'
    adaptive_sample = 'SAMPLE_RNN ADAPTIVE'

    baseline_series = [phased_rnn, rnn, sample_rnn, skip_rnn]

    # Create the table headers
    table_lines: List[str] = []
    table_lines.append('\\begin{tabular}{lccccc}')

    headers: List[str] = ['\\textbf{Dataset}']
    for baseline in baseline_series:
        h = '\\textbf{{ {0} }}'.format(to_label(baseline))
        headers.append(h)

    table_lines.append(' & '.join(headers))
    table_lines.append('\\midrule')

    all_greater: Counter() = Counter()
    all_less: Counter() = Counter()

    for dataset_name, dataset_results in sorted(results.items()):
        dataset_label = rename_dataset(dataset_name)

        greater_than_adaptive = Counter()
        less_than_adaptive = Counter()

        greater_than_acc = defaultdict(list)
        less_than_acc = defaultdict(list)

        for budget, model_results in dataset_results.items():

            adaptive_acc = np.average([e.accuracy for e in model_results[adaptive_sample]])

            for baseline in baseline_series:
                baseline_acc = np.average([e.accuracy for e in model_results[baseline]])

                if baseline_acc >= adaptive_acc:
                    greater_than_adaptive[baseline] += 1
                    greater_than_acc[baseline].append(baseline_acc - adaptive_acc)
                    all_greater[baseline] += 1
                else:
                    less_than_adaptive[baseline] += 1
                    less_than_acc[baseline].append(adaptive_acc - baseline_acc)
                    all_less[baseline] += 1

        line: List[str] = [dataset_label]
        for baseline in baseline_series:
            less = less_than_adaptive[baseline]
            greater = greater_than_adaptive[baseline]

            if less > greater:
                comparison = '\\textbf{{{0}}} / {1}'.format(less, greater)
            elif less < greater:
                comparison = '{0} / \\textbf{{{1}}}'.format(less, greater)
            else:
                comparison = '{0} / {1}'.format(less, greater)

            line.append(comparison)

        table_lines.append(' & '.join(line))

    # Add ending
    table_lines.append('\\end{tabular}')

    # Print Latex table
    print('\\\\\n'.join(table_lines))

    for baseline in baseline_series:
        num = all_less[baseline]
        denom = all_greater[baseline] + all_less[baseline]
        print('Improvement Percentage over {0}: {1:.4f} ({2} / {3})'.format(baseline, num / denom, num, denom))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folders', type=str, required=True, nargs='+')
    parser.add_argument('--noise-loc', type=float, required=True)
    parser.add_argument('--noise-scale', type=float, required=True)
    parser.add_argument('--baseline-mode', type=str, required=True)
    args = parser.parse_args()

    noise_generator = make_noise_generator(noise_type='gaussian',
                                           noise_loc=args.noise_loc,
                                           noise_scale=args.noise_scale,
                                           noise_period=None,
                                           noise_amplitude=None)

    all_results = get_results(args.input_folders, noise_generator, model_type='rnn', baseline_mode=args.baseline_mode)

    make_table(all_results, baseline_mode=args.baseline_mode)
