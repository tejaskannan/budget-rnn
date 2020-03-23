import scipy.stats
import os
import re
import numpy as np
from argparse import ArgumentParser
from typing import List, Dict, Any
from itertools import combinations

from utils.file_utils import read_by_file_suffix, make_dir, save_by_file_suffix


MODEL_TYPE_REGEX = re.compile(r'.*model-test-log-([^-]+)-.*')
MODEL = 'model'
SCHEDULED_MODEL = 'scheduled_model'
ALL_LATENCY = 'ALL_LATENCY'


def get_model_type(test_log_file: str) -> str:
    match = MODEL_TYPE_REGEX.match(test_log_file)
    return match.group(1)


def run_tests(test_log_files: List[str], output_file: str):
    # Fetch latency measurements
    latencies: Dict[str, List[float]] = dict()
    for test_log_file in test_log_files:
        test_log = list(read_by_file_suffix(test_log_file))[0]
        model_type = get_model_type(test_log_file)

        if MODEL in test_log:
            latencies[model_type] = test_log[MODEL][ALL_LATENCY]
        elif SCHEDULED_MODEL in test_log:
            latencies[model_type] = test_log[SCHEDULED_MODEL][ALL_LATENCY]

    # List to hold test results
    results: List[Dict[str, Any]] = []

    # Run latency tests
    model_types = list(sorted(latencies.keys()))
    for first_type, second_type in combinations(model_types, 2):
        first_latencies, second_latencies = latencies[first_type], latencies[second_type]
        t_stat, p_value = scipy.stats.ttest_ind(first_latencies, second_latencies, equal_var=False)

        result_dict = dict(first_type=first_type,
                           second_type=second_type,
                           first_avg=np.average(first_latencies),
                           second_avg=np.average(second_latencies),
                           first_std=np.std(first_latencies),
                           second_std=np.std(second_latencies),
                           t_stat=t_stat,
                           p_value=p_value)

        results.append(result_dict)

    # Save results
    save_by_file_suffix(results, output_file)


if __name__ == '__main__':
    parser = ArgumentParser('Script to run pairwise Welch\'s t-tests on inference latency.')
    parser.add_argument('--test-logs', type=str, nargs='+')
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()

    assert len(args.test_logs) > 1, 'Must have at least 2 test logs.'

    run_tests(args.test_logs, args.output_file)
