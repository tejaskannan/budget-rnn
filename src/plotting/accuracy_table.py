import numpy as np
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from typing import Dict, List, Set, DefaultDict

from plotting_utils import ModelResult, get_results, make_noise_generator, to_label, rename_dataset, get_model_name


DatasetResult = namedtuple('DatasetResult', ['mean', 'std', 'maximum', 'minimum', 'first', 'third'])
NEWLINE = '\\\\'


def create_table(model_results: Dict[str, Dict[str, DatasetResult]], models_to_keep: List[str]):
    """
    Formats a Latex table with the given model results.

    Args:
        model_results: Dictionary mapping dataset -> (dict of model -> accuracy)
    """
    rows: List[str] = []
    rows.append('\\begin{tabular}{lccc}')

    headers = ['Dataset'] + list(models_to_keep)
    headers = list(map(lambda t: '\\textbf{{ {0} }}'.format(to_label(t)), headers))
    header = ' & '.join(headers)

    rows.append(header + NEWLINE)
    rows.append('\\midrule')

    # Format the results from each data-set
    acc_dict: DefaultDict[str, List[float]] = defaultdict(list)

    for dataset, results in sorted(model_results.items()):
        row: List[str] = [dataset]

        for model in models_to_keep:
            assert model in results, 'Dataset {0} does not have {1}'.format(dataset, model)

            accuracy = results[model]
            row.append('{0:.3f}'.format(accuracy.mean))

            acc_dict[model].append(accuracy.mean)

        rows.append(' & '.join(row) + NEWLINE)

    rows.append('\\midrule')

    # Compute the total average over all data-sets
    row = ['All']
    for model in models_to_keep:
        row.append('{0:.3f}'.format(np.average(acc_dict[model])))

    rows.append(' & '.join(row))

    rows.append('\\end{tabular}')

    return '\n'.join(rows)


def merge_results(model_results: Dict[str, DefaultDict[float, Dict[str, List[ModelResult]]]],
                  models_to_keep: List[str],
                  baseline_name: str) -> Dict[str, Dict[str, DatasetResult]]:

    merged: Dict[str, DefaultDict[str, List[float]]] = dict()

    for dataset, dataset_results in model_results.items():
        dataset_name = rename_dataset(dataset)

        merged[dataset_name] = defaultdict(list)

        for budget, results in dataset_results.items():
            if budget > 3:
                continue

            baseline_acc = np.average([r.accuracy for r in results[baseline_name]])

            for model_name in models_to_keep:
                # Ensure the model name is valid
                assert model_name in results, 'Could not find {0} in {1}'.format(model_name, list(results.keys()))

                accuracy = np.average([r.accuracy for r in results[model_name]])

                merged[dataset_name][model_name].append(accuracy - baseline_acc)

    aggregated: Dict[str, Dict[str, DatasetResult]] = dict()
    for dataset_name, merged_results in merged.items():

        dataset_aggregate = dict()
        for model_name in models_to_keep:
            model_accuracy = np.array(merged_results[model_name])
            model_aggregate = DatasetResult(mean=np.average(model_accuracy),
                                            std=np.std(model_accuracy),
                                            maximum=np.max(model_accuracy),
                                            minimum=np.min(model_accuracy),
                                            first=np.percentile(model_accuracy, 25),
                                            third=np.percentile(model_accuracy, 75))
            dataset_aggregate[model_name] = model_aggregate

        aggregated[dataset_name] = dataset_aggregate

    return aggregated


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folders', type=str, nargs='+', required=True)
    parser.add_argument('--mode', type=str, choices=['baseline', 'sample'], required=True)
    parser.add_argument('--noise-loc', type=float, default=0.0)
    args = parser.parse_args()

    noise = make_noise_generator(noise_type='gaussian',
                                 noise_loc=args.noise_loc,
                                 noise_scale=0.01,
                                 noise_period=None,
                                 noise_amplitude=None)

    model_results = get_results(input_folders=args.input_folders,
                                model_type='rnn',
                                noise_generator=noise,
                                baseline_mode='under_budget')

    baseline_name = 'RNN FIXED_UNDER_BUDGET'

    if args.mode == 'sample':
        to_keep = ['SAMPLE_RNN FIXED_UNDER_BUDGET', 'SAMPLE_RNN RANDOMIZED', 'SAMPLE_RNN ADAPTIVE']
    else:
        to_keep = ['PHASED_RNN FIXED_UNDER_BUDGET', 'SKIP_RNN FIXED_UNDER_BUDGET', 'SAMPLE_RNN ADAPTIVE']

    merged = merge_results(model_results=model_results,
                           baseline_name=baseline_name,
                           models_to_keep=to_keep)

    print(create_table(merged, models_to_keep=to_keep))
