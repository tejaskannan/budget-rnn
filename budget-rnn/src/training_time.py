"""
This script computer the training time for various RNN models.
For Budget RNNs, this time includes the halting threshold optimization.
"""
import re
from argparse import ArgumentParser
from datetime import datetime, timedelta
from typing import List

from utils.file_utils import iterate_files, read_by_file_suffix


TIME_REGEX = re.compile(r'.*model-train-log-[^0-9]+([0-9\-]+)_model_best\.pkl\.gz')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True, help='Folder containing RNN models (of the same type) to measure.')
    args = parser.parse_args()

    total_time = timedelta()

    times_list: List[datetime] = []
    model_count = 0

    for train_log_path in iterate_files(args.input_folder, pattern=r'.*model-train-log-.*\.pkl\.gz'):
        train_log = read_by_file_suffix(train_log_path)

        if 'start_time' not in train_log:
            match = TIME_REGEX.match(train_log_path)
            start_date = datetime.strptime(match.group(1), '%Y-%m-%d-%H-%M-%S')

            times_list.append(start_date)
        else:

            start_time, end_time = train_log['start_time'], train_log['end_time']

            start_date = datetime.strptime(start_time, '%Y-%m-%d-%H-%M-%S')
            end_date = datetime.strptime(end_time, '%Y-%m-%d-%H-%M-%S')

            time_delta = (end_date - start_date)
            total_time += time_delta

        model_count += 1

    if len(times_list) > 0:
        n = len(times_list)

        for i in range(n - 1):
            total_time += (times_list[i+1] - times_list[i])

        total_time = (n / (n - 1)) * total_time

    for controller_path in iterate_files(args.input_folder, pattern=r'.*model-controller-bluetooth-.*\.pkl\.gz'):
        controller_results = read_by_file_suffix(controller_path)

        start_time, end_time = controller_results['fit_start_time'], controller_results['fit_end_time']

        start_date = datetime.strptime(start_time, '%Y-%m-%d-%H-%M-%S')
        end_date = datetime.strptime(end_time, '%Y-%m-%d-%H-%M-%S')

        time_delta = (end_date - start_date)
        total_time += time_delta

    # Convert to minutes
    total_minutes = total_time.total_seconds() / 60

    print('Total Time: {0:.1f} minutes'.format(total_minutes))
    print('Number of models: {0}'.format(model_count))
