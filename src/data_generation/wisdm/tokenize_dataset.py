"""
Tokenizes the WISDM human activity recognition dataset. The original dataset can be found
here: http://www.cis.fordham.edu/wisdm/dataset.php
"""


import os.path
from argparse import ArgumentParser
from collections import Counter, deque
from typing import Iterable, Dict, Any, List
from copy import deepcopy

from utils.data_writer import DataWriter
from utils.file_utils import make_dir
from utils.constants import INPUTS, OUTPUT, SAMPLE_ID, TIMESTAMP
from utils.constants import TRAIN, VALID, TEST


LABEL_MAP = {
    'walking': 0,
    'jogging': 1,
    'sitting': 2,
    'standing': 3,
    'upstairs': 4,
    'downstairs': 5
}


def get_partition(user_id: int) -> str:
    if user_id <= 19:
        return TRAIN  # [0, 19]
    elif user_id <= 26:
        return VALID  # [20, 26]
    elif user_id <= 36:
        return TEST  # [27, 36]
    else:
        raise ValueError('Unexpected user id {0}'.format(user_id))


def get_data_generator(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, 'r') as input_file:
        for line in input_file:
            tokens = [t.replace(';', '').strip() for t in line.split(',')]
            tokens = [t for t in tokens if len(t) > 0]

            # Skip incomplete samples
            if len(tokens) < 6:
                continue

            sample_dict = {
                INPUTS: [float(tokens[3]), float(tokens[4]), float(tokens[5])],
                OUTPUT: LABEL_MAP[tokens[1].lower()],
                TIMESTAMP: int(tokens[2]),
                SAMPLE_ID: int(tokens[0])  # The user id
            }

            yield sample_dict


def majority(labels: List[int]) -> int:
    counter: Counter = Counter()
    for label in labels:
        counter[label] += 1

    return counter.most_common(1)[0][0]


def tokenize(input_path: str, output_folder: str, window: int, stride: int, chunk_size: int):
    
    # Create the data writers
    make_dir(output_folder)
    data_writers = {
        TRAIN: DataWriter(os.path.join(output_folder, TRAIN), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size),
        VALID: DataWriter(os.path.join(output_folder, VALID), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size),
        TEST: DataWriter(os.path.join(output_folder, TEST), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size)
    }

    # Count the label distribution for each fold
    label_counter = {
        TRAIN: Counter(),
        VALID: Counter(),
        TEST: Counter()
    }

    data_generator = get_data_generator(input_path)
    data_window = deque()

    stride_counter = stride
    sample_id = 0
    current_label, current_subject = None, None
    for sample in data_generator:
        if len(data_window) == 0 or current_label is None or current_subject is None:
            data_window.append(sample)
            current_label = sample[OUTPUT]
            current_subject = sample[SAMPLE_ID]
        elif sample[OUTPUT] != current_label or sample[SAMPLE_ID] != current_subject:  # Reset the queue if the label or subject changes
            data_window.clear()
            data_window.append(sample)
            current_label = sample[OUTPUT]
            current_subject = sample[SAMPLE_ID]
        else:
            while len(data_window) >= window:
                data_window.popleft()

            data_window.append(sample)

        # Skip samples corresponding to the stride
        stride_counter += 1
        if stride_counter < stride:
            continue

        # Don't write incomplete samples
        if len(data_window) < window:
            continue

        inputs = [element[INPUTS].copy() for element in data_window]

        data_sample = {
            SAMPLE_ID: sample_id,
            INPUTS: inputs,
            OUTPUT: current_label,
            TIMESTAMP: data_window[-1][TIMESTAMP]
        }

        partition = get_partition(current_subject)
        data_writers[partition].add(data_sample)

        label_counter[partition][current_label] += 1

        stride_counter = 0
        sample_id += 1

        if (sample_id + 1) % chunk_size == 0:
            print('Completed {0} samples'.format(sample_id + 1), end='\r')
    print()

    # Close all writers
    for writer in data_writers.values():
        writer.close()

    for fold, counter in label_counter.items():
        print('{0}:'.format(fold))
        total = sum(counter.values())
        for label, count in sorted(counter.items()):
            print('{0}: {1} ({2:.3f})'.format(label, count, count / total))


if __name__ == '__main__':
    parser = ArgumentParser('Script to tokenize the HAR dataset.')
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--stride', type=int, required=True)
    parser.add_argument('--chunk-size', type=int, default=10000)
    args = parser.parse_args()

    assert args.window > 0, 'Must have a positive window'
    assert args.stride > 0, 'Must have a positive stride'

    assert os.path.exists(args.input_file), 'The file {0} does not exist!'.format(args.input_file)

    tokenize(input_path=args.input_file,
             output_folder=args.output_folder,
             window=args.window,
             stride=args.stride,
             chunk_size=args.chunk_size)
