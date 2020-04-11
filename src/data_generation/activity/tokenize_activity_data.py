import os.path
from argparse import ArgumentParser
from typing import Iterable, Dict, Any, List
from random import random
from collections import Counter

from utils.constants import INPUTS, SAMPLE_ID, OUTPUT, SMALL_NUMBER, TIMESTAMP
from utils.file_utils import iterate_files, read_by_file_suffix
from utils.data_writer import DataWriter


LABEL = 'label'
LABELS = 'labels'
FREQ = 0.01  # This is a parameter of the dataset


def from_sensor(label: str, sensors: List[str]) -> bool:
    for sensor in sensors:
        if label.startswith(sensor):
            return True
    return False


def get_data_iterator(input_folder: str, sensors: List[str]) -> Iterable[Dict[str, Any]]:
    data_files = iterate_files(input_folder, pattern='.*jsonl.gz')
    for data_file in data_files:
        for sample in read_by_file_suffix(data_file):
            data_dict: Dict[str, Any] = dict()
            data_dict[OUTPUT] = sample[LABEL]
            data_dict[INPUTS] = [val for key, val in sorted(sample.items()) if from_sensor(key, sensors)]
            data_dict[TIMESTAMP] = sample[TIMESTAMP]
            yield data_dict


def majority(labels: List[int]) -> int:
    label_counter: Counter = Counter()
    for label in labels:
        label_counter[label] += 1
    return label_counter.most_common(1)[0][0]


def tokenize_data(input_folder: str,
                  output_folder: str,
                  window: int,
                  stride: int,
                  skip_labels: List[int],
                  sensors: List[str],
                  file_prefix: str,
                  chunk_size: int,
                  sample_frac: float):
    """
    Function to tokenize activity datasets.
    """    
    data_iterator = get_data_iterator(input_folder, sensors=sensors)

    with DataWriter(output_folder, file_prefix=file_prefix, file_suffix='jsonl.gz', chunk_size=chunk_size) as writer:

        sample_id = 0
        data_window: List[Dict[str, Any]] = []
        stride_counter = stride
        label_counter = Counter()

        for data_index, data_sample in enumerate(data_iterator):
            # Skip data according to the stride policy
            if stride_counter < stride:
                stride_counter += 1
                continue

            # Create windows of sufficient size and validate timestamps
            if len(data_window) == 0:
                data_window.append(data_sample)
            elif abs(data_sample[TIMESTAMP] - data_window[-1][TIMESTAMP]) > FREQ + SMALL_NUMBER:
                data_window = [data_sample]
            else:
                data_window.append(data_sample)

            if len(data_window) == window:
                labels = [elem[OUTPUT] for elem in data_window]
                label = majority(labels)

                element = {
                    SAMPLE_ID: sample_id,
                    INPUTS: [sample[INPUTS] for sample in data_window],
                    OUTPUT: label,
                    LABELS: labels,
                    TIMESTAMP: data_window[-1][TIMESTAMP]
                }

                # Perform sample filtering
                r = random()
                if (skip_labels is None or label not in skip_labels) and r < sample_frac:
                    writer.add(element)
                    sample_id += 1
                    stride_counter = 0
                    label_counter[label] += 1

                # Reset the data window
                data_window = []

            if (sample_id + 1) % chunk_size == 0:
                print(f'Completed {data_index + 1} samples.', end='\r')

        print()
        print(f'Completed processing. Total of {sample_id + 1} samples')
        print('Label Distribution:')
        for key, val in sorted(label_counter.items()):
            frac = val / sample_id
            print(f'{key}: {val} ({frac:.3f})')


if __name__ == '__main__':
    parser = ArgumentParser('Script to create tokenized activity datasets.')
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--stride', type=int, required=True)
    parser.add_argument('--skip-labels', type=int, nargs='*')
    parser.add_argument('--sensors', type=str, nargs='+', choices=['hand', 'chest', 'ankle'])
    parser.add_argument('--file-prefix', type=str, default='data')
    parser.add_argument('--chunk-size', type=int, default=5000)
    parser.add_argument('--sample-frac', type=float, default=1.0)
    args = parser.parse_args()

    # Validate arguments
    assert os.path.exists(args.input_folder), f'The folder {args.input_folder} does not exist!'
    assert 0 < args.sample_frac and args.sample_frac <= 1.0, 'The sample fraction must be in the range (0, 1]'
    assert args.window > 0, 'Must have a positive window size'
    assert args.stride > 0, 'Must have a positive stride'

    tokenize_data(input_folder=args.input_folder,
                  output_folder=args.output_folder,
                  window=args.window,
                  stride=args.stride,
                  skip_labels=args.skip_labels,
                  sensors=args.sensors,
                  file_prefix=args.file_prefix,
                  chunk_size=args.chunk_size,
                  sample_frac=args.sample_frac)
