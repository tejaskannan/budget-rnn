import os.path
from argparse import ArgumentParser
from collections import Counter, deque
from typing import Iterable, Dict, Any, List
from copy import deepcopy

from utils.data_writer import DataWriter
from utils.constants import INPUTS, OUTPUT, SAMPLE_ID, TIMESTAMP


LABEL_MAP = {
    'walking': 0,
    'jogging': 1,
    'sitting': 2,
    'standing': 3,
    'upstairs': 4,
    'downstairs': 5
}


def get_data_generator(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, 'r') as input_file:
        for line in input_file:
            tokens = [t.replace(';', '').strip() for t in line.split(',')]
            tokens = [t for t in tokens if len(t) > 0]

            # Skip incomplete samples
            if len(tokens) < 6:
                continue

            try:
                sample_dict = {
                    INPUTS: [float(tokens[3]), float(tokens[4]), float(tokens[5])],
                    OUTPUT: LABEL_MAP[tokens[1].lower()],
                    TIMESTAMP: int(tokens[2])
                }

                yield sample_dict
            except ValueError:
                raise


def majority(labels: List[int]) -> int:
    counter: Counter = Counter()
    for label in labels:
        counter[label] += 1

    return counter.most_common(1)[0][0]


def tokenize(input_path: str, output_folder: str, window: int, stride: int, chunk_size: int):

    with DataWriter(output_folder, file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size) as writer:
        
        data_generator = get_data_generator(input_path)
        data_window = deque()
        label_counter = Counter()
        
        stride_counter = stride
        sample_id = 0
        for sample in data_generator:
            data_window.append(sample)
            if len(data_window) < window:
                continue

            # Remove excess elements
            while len(data_window) > window:
                data_window.popleft()

            stride_counter += 1
            if stride_counter < stride:
                continue

            labels = [element[OUTPUT] for element in data_window]
            label = majority(labels)
            label_counter[label] += 1

            inputs = [element[INPUTS].copy() for element in data_window]

            data_sample = {
                SAMPLE_ID: sample_id,
                INPUTS: inputs,
                OUTPUT: label,
                TIMESTAMP: data_window[-1][TIMESTAMP]
            }
            writer.add(data_sample)

            stride_counter = 0
            sample_id += 1

            if (sample_id + 1) % chunk_size == 0:
                print('Completed {0} samples'.format(sample_id + 1), end='\r')
    print()

    total = sum(label_counter.values())
    for label, count in sorted(label_counter.items()):
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
