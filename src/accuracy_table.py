import numpy as np
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from typing import Dict, List, Set, DefaultDict

from plotting.plotting_utils import ModelResult, get_results, make_noise_generator, to_label, rename_dataset, get_model_name


DatasetResult = namedtuple('DatasetResult', ['mean', 'std', 'maximum', 'minimum', 'first', 'third', 'geom', 'raw'])
NEWLINE = '\\\\'


def geometric_mean(values: np.ndarray) -> float:
    prod = np.prod(values)
    return np.power(prod, (1.0 / len(values)))


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
            if model not in results:
                continue

            accuracy = results[model]
            row.append('{0:.3f}'.format(accuracy.geom))

            acc_dict[model].append(accuracy.raw)

        rows.append(' & '.join(row) + NEWLINE)

    rows.append('\\midrule')

    # Compute the total average over all data-sets
    row = ['All']
    for model in models_to_keep:
        if model not in acc_dict:
            continue

        accuracy_values = np.concatenate(acc_dict[model])
        mean = geometric_mean(accuracy_values)
        row.append('{0:.3f}'.format(mean))

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
            baseline_acc = np.average([r.accuracy for r in results[baseline_name]])

            for model_name in models_to_keep:
                if model_name not in results:
                    continue

                assert len(results[model_name]) == 1, 'Only supports 1 trial per model'
            
                accuracy = results[model_name][0].accuracy
                
                # merged[dataset_name][model_name].append(accuracy - baseline_acc)
                merged[dataset_name][model_name].append(accuracy)


    aggregated: Dict[str, Dict[str, DatasetResult]] = dict()
    for dataset_name, merged_results in merged.items():

        dataset_aggregate = dict()
        for model_name in models_to_keep:
            if model_name not in merged_results:
                continue

            model_accuracy = np.array(merged_results[model_name])

            model_aggregate = DatasetResult(mean=np.average(model_accuracy),
                                            std=np.std(model_accuracy),
                                            maximum=np.max(model_accuracy),
                                            minimum=np.min(model_accuracy),
                                            first=np.percentile(model_accuracy, 25),
                                            third=np.percentile(model_accuracy, 75),
                                            geom=geometric_mean(model_accuracy),
                                            raw=model_accuracy)
            dataset_aggregate[model_name] = model_aggregate

        aggregated[dataset_name] = dataset_aggregate

    return aggregated


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folders', type=str, nargs='+', required=True, help='Paths to folders containing the merged logs. These may be from multiple datsets.')
    parser.add_argument('--mode', type=str, choices=['baseline', 'budget'], required=True, help='The comparison mode. Either comparison to baselines or comparison to Budget RNN versions.')
    parser.add_argument('--noise-loc', type=float, default=0.0, help='The noise bias.')
    args = parser.parse_args()

    noise = make_noise_generator(noise_type='gaussian',
                                 noise_loc=args.noise_loc,
                                 noise_scale=0.05,
                                 noise_period=None,
                                 noise_amplitude=None)

    model_results = get_results(input_folders=args.input_folders,
                                model_type='rnn',
                                noise_generator=noise)

    baseline_name = 'RNN FIXED_UNDER_BUDGET'

    if args.mode == 'budget':
        to_keep = ['SAMPLE_RNN FIXED_UNDER_BUDGET', 'SAMPLE_RNN RANDOMIZED', 'SAMPLE_RNN ADAPTIVE', 'BUDGET_RNN FIXED_UNDER_BUDGET', 'BUDGET_RNN RANDOMIZED', 'BUDGET_RNN ADAPTIVE']
    else:
        to_keep = ['RNN FIXED_UNDER_BUDGET', 'PHASED_RNN FIXED_UNDER_BUDGET', 'SKIP_RNN FIXED_UNDER_BUDGET', 'SAMPLE_RNN ADAPTIVE', 'BUDGET_RNN ADAPTIVE']

    merged = merge_results(model_results=model_results,
                           baseline_name=baseline_name,
                           models_to_keep=to_keep)

    print(create_table(merged, models_to_keep=to_keep))
