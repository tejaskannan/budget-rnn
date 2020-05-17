import os.path
from argparse import ArgumentParser
from collections import Counter, deque
from typing import Iterable, Dict, Any, List
from copy import deepcopy

from utils.data_writer import DataWriter
from utils.constants import INPUTS, OUTPUT, SAMPLE_ID
from utils.file_utils import read_by_file_suffix


def get_data_iterator(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, 'r') as data_file:

        for line in data_file:
            tokens = line.split(',')

            try:
                features = [float(val) for val in tokens[:-1]]
                sample = {
                    INPUTS: features,
                    OUTPUT: int(tokens[-1])
                }

                yield sample
            except ValueError:
                raise


def majority(elements: List[int]) -> int:
    elem_counter: Counter = Counter()
    for x in elements:
        elem_counter[x] += 1
    return elem_counter.most_common(1)[0][0]


def tokenize_data(input_file: str, output_folder: str, window: int, stride: int, chunk_size: int):

    with DataWriter(output_folder, file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size) as writer:
        
        label_counter = Counter()
        data_window: deque = deque()

        stride_counter = stride
        sample_id = 0
        for sample in get_data_iterator(input_file):
            data_window.append(sample)
            
            # Remove excess data entries
            while len(data_window) > window:
                data_window.popleft()
            
            stride_counter += 1

            # Only write when the stride is fully reached
            if stride_counter >= stride and len(data_window) == window: 

                label = majority([elem[OUTPUT] for elem in data_window])
                sample_dict = {
                    SAMPLE_ID: sample_id,
                    INPUTS: [deepcopy(elem[INPUTS]) for elem in data_window],
                    OUTPUT: label
                }

                writer.add(sample_dict)
                stride_counter = 0
                sample_id += 1
                label_counter[label] += 1

        print(label_counter)


if __name__ == '__main__':
    parser = ArgumentParser('Script to tokenize Eye sensor dataset.')
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--stride', type=int, required=True)
    parser.add_argument('--chunk-size', type=int, required=True)
    args = parser.parse_args()

    tokenize_data(input_file=args.input_file,
                  output_folder=args.output_folder,
                  window=args.window,
                  stride=args.stride,
                  chunk_size=args.chunk_size)
