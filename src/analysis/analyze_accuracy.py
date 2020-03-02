import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from argparse import ArgumentParser
from collections import defaultdict
from dpu_utils.utils import RichPath
from typing import Dict, Any, List, DefaultDict, Tuple


WIDTH = 0.35


def get_aggregate_stats(stats_dict: Dict[str, Dict[str, List[float]]]) -> DefaultDict[str, Dict[str, float]]:

    agg_stats: DefaultDict[str, Dict[str, float]] = defaultdict(dict)

    for op_name, accuracies in stats_dict['accuracy'].items():
        agg_stats[op_name]['accuracy'] = float(np.average(accuracies))

    for op_name, latencies in stats_dict['latency'].items():
        agg_stats[op_name]['latency_mean'] = float(np.average(latencies))
        agg_stats[op_name]['latency_std'] = float(np.std(latencies))

    for op_name, precision in stats_dict['precision'].items():
        agg_stats[op_name]['precision'] = float(np.average(precision))

    for op_name, recall in stats_dict['recall'].items():
        agg_stats[op_name]['recall'] = float(np.average(recall))

    return agg_stats


def latency_ttest(stats_dict: Dict[str, Dict[str, List[float]]]) -> DefaultDict[Tuple[str, str], Dict[str, float]]:
    
    latency_tests: List[Dict[str, float]] = []

    completed_pairs = set()
    for first_op in stats_dict['latency'].keys():
        for second_op in stats_dict['latency'].keys():
            pair = (first_op, second_op) if first_op < second_op else (second_op, first_op)

            if pair in completed_pairs or first_op == second_op:
                continue

            first_latencies = stats_dict['latency'][pair[0]]
            second_latencies = stats_dict['latency'][pair[1]]
            t_stat, p_value = scipy.stats.ttest_ind(first_latencies, second_latencies, equal_var=False)

            latency_tests.append(dict(op_one=pair[0], op_two=pair[1], t_stat=t_stat, p_value=p_value))

            completed_pairs.add(pair)

    return latency_tests


def plot(agg_stats: DefaultDict[str, Dict[str, float]], metric: str, output_folder: RichPath):
    with plt.style.context('ggplot'):
        
        fig, ax = plt.subplots(figsize=(12, 9))

        labels = list(sorted(agg_stats.keys()))
        for i, label in enumerate(labels):
            ax.bar(label, agg_stats[label][metric], width=WIDTH, align='center', label=label)

            ax.annotate('{0:.4f}'.format(agg_stats[label][metric]), (i, agg_stats[label][metric]), textcoords='offset points', xytext=(0, 5), ha='center')


        name = metric.capitalize()
        ax.set_title('{0} Results for Each Model Type'.format(name))
        ax.set_ylabel(name)
        ax.set_xlabel('Model Type')

        output_file = output_folder.join('{0}.pdf'.format(metric)).path
        plt.savefig(output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--accuracy-file', type=str, required=True)
    args = parser.parse_args() 

    accuracy_file = RichPath.create(args.accuracy_file)
    assert accuracy_file.exists(), f'The file {accuracy_file} does not exist!'

    output_folder = RichPath.create(accuracy_file.path[:-len('.jsonl.gz')])
    output_folder.make_as_dir()

    stats_dict = list(accuracy_file.read_by_file_suffix())[0]

    agg_stats = get_aggregate_stats(stats_dict)

    agg_stats_file = output_folder.join('aggregate_stats.jsonl.gz')
    agg_stats_file.save_as_compressed_file([agg_stats])

    latency_test_results = latency_ttest(stats_dict)
    latency_test_file = output_folder.join('latency_ttests.jsonl.gz')
    latency_test_file.save_as_compressed_file([latency_test_results])

    plot(agg_stats, metric='recall', output_folder=output_folder)
    plot(agg_stats, metric='precision', output_folder=output_folder)
    plot(agg_stats, metric='accuracy', output_folder=output_folder)
