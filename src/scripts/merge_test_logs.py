from argparse import ArgumentParser
from typing import Dict, Any, List, DefaultDict
from collections import defaultdict

from utils.file_utils import read_by_file_suffix, save_by_file_suffix
from utils.testing_utils import ClassificationMetric


def merge_logs(test_logs: List[Dict[str, Any]], output_file: str):
    
    output_names = list(sorted(test_logs[0].keys()))

    merged_log: Dict[str, DefaultDict[str, List[float]]] = dict()
    for output_name in output_names:
        
        output_log: DefaultDict[str, List[float]] = defaultdict(list)
        for log in test_logs:
            for metric in ClassificationMetric:
                output_log[metric.name].append(log[output_name][metric.name])

        merged_log[output_name] = output_log

    save_by_file_suffix([merged_log], output_file)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-logs', type=str, nargs='+')
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()

    test_logs: List[Dict[str, Any]] = []
    for test_log_path in args.test_logs:
        test_log = list(read_by_file_suffix(test_log_path))[0]
        test_logs.append(test_log)

    merge_logs(test_logs, args.output_file)
