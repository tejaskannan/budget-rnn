import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from argparse import ArgumentParser
from collections import defaultdict
from dpu_utils.utils import RichPath
from typing import Dict, Any, List, DefaultDict, Tuple

from utils.testing_utils import ClassificationMetric
from utils.constants import SMALL_NUMBER

WIDTH = 0.35


def get_aggregate_stats(stats_dict: Dict[str, Dict[str, List[float]]]) -> DefaultDict[str, Dict[str, float]]:

    agg_stats: DefaultDict[str, Dict[str, float]] = defaultdict(dict)

    for op_name, metric_dict in stats_dict.items():
        for metric_name in ClassificationMetric:
            # Remove nonzero values from precision and recall
            metric_values = metric_dict[metric_name.name]
            if metric_name in (ClassificationMetric.PRECISION, ClassificationMetric.RECALL):
                metric_values = [x for x in metric_values if abs(x) > SMALL_NUMBER]
            elif metric_name == ClassificationMetric.LATENCY:
                metric_values = metric_values[1:]
            
            agg_stats[op_name][metric_name.name + '_mean'] = float(np.average(metric_values))
            agg_stats[op_name][metric_name.name + '_std'] = float(np.std(metric_values))

    return agg_stats


def t_tests(stats_dict: Dict[str, Dict[str, List[float]]], metric: ClassificationMetric, output_folder: RichPath):
    
    test_results: List[Dict[str, float]] = []

    completed_pairs = set()
    for first_op, first_dict in stats_dict.items():
        for second_op, second_dict in stats_dict.items():
            pair = (first_op, second_op) if first_op < second_op else (second_op, first_op)

            if pair in completed_pairs or first_op == second_op:
                continue

            first_values = stats_dict[pair[0]][metric.name]
            second_values = stats_dict[pair[1]][metric.name]
            if metric in (ClassificationMetric.PRECISION, ClassificationMetric.RECALL):
                first_values = [x for x in first_values if abs(x) > SMALL_NUMBER]
                second_values = [x for x in second_values if abs(x) > SMALL_NUMBER]
            elif metric == ClassificationMetric.LATENCY:
                first_values = first_values[1:]
                second_values = second_values[1:]

            t_stat, p_value = scipy.stats.ttest_ind(first_values, second_values, equal_var=False)

            test_results.append(dict(op_one=pair[0], op_two=pair[1], t_stat=t_stat, p_value=p_value))

            completed_pairs.add(pair)

    output_file = output_folder.join(metric.name.lower() + '-t-test.jsonl.gz')
    output_file.save_as_compressed_file(test_results)


def plot(agg_stats: DefaultDict[str, Dict[str, float]], metric: ClassificationMetric, output_folder: RichPath):
    with plt.style.context('ggplot'):
        
        fig, ax = plt.subplots(figsize=(12, 9))

        labels = list(sorted(agg_stats.keys()))
        for i, label in enumerate(labels):
            metric_avg_name = metric.name + '_mean'
            metric_std_name = metric.name + '_std'
            
            avg_value = agg_stats[label][metric_avg_name]
            ax.bar(label, avg_value, width=WIDTH, align='center', label=label)

            ax.annotate('{0:.4f}'.format(avg_value), (i, avg_value), textcoords='offset points', xytext=(0, 5), ha='center')

        name = metric.name.lower().capitalize()
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

    #latency_test_results = latency_ttest(stats_dict)
    #latency_test_file = output_folder.join('latency_ttests.jsonl.gz')
    #latency_test_file.save_as_compressed_file([latency_test_results])

    for metric in ClassificationMetric:
        t_tests(stats_dict, metric=metric, output_folder=output_folder)
        plot(agg_stats, metric=metric, output_folder=output_folder)
