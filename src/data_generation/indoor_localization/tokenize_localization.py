import os.path
import re
import numpy as np
from argparse import ArgumentParser
from collections import Counter, deque
from typing import Iterable, Dict, Any, List

from utils.data_writer import DataWriter
from utils.file_utils import make_dir, iterate_files
from utils.constants import INPUTS, OUTPUT, SAMPLE_ID, TIMESTAMP, TRAIN, VALID, TEST


LABEL_MAP = {
    -1: 0,
    1: 1
}

DATA_FILE_REGEX = re.compile(r'.*_([0-9]+)\.csv')
ORIGINAL_ID = 'original_id'


def load_class_labels(path: str) -> Dict[int, int]:
    with open(path, 'r') as label_file:
        file_iterator = iter(label_file)
        next(file_iterator)  # Skip the headings

        class_label_map: Dict[int, int] = dict()
        for line in file_iterator:
            tokens = line.split(',')

            seq_id = int(tokens[0].strip())
            label = LABEL_MAP[int(tokens[1].strip())]
    
            class_label_map[seq_id] = label

    return class_label_map


def load_partition_map(path: str) -> Dict[int, int]:
    with open(path, 'r') as partition_file:
        file_iterator = iter(partition_file)
        next(file_iterator)  # Skip the headings

        partition_map: Dict[int, int] = dict()
        for line in file_iterator:
            tokens = line.split(',')

            seq_id = int(tokens[0].strip())
            partition = int(tokens[1].strip())
    
            partition_map[seq_id] = partition

    return partition_map


def get_data_generator(path: str, window: int, reps: int, label_map: Dict[int, int]) -> Iterable[Dict[str, Any]]:

    for data_file_path in iterate_files(path, pattern=r'.*csv'):
        id_match = DATA_FILE_REGEX.match(data_file_path)
        if id_match is None:
            continue

        file_id = int(id_match.group(1))

        features: List[List[float]] = []
        with open(data_file_path, 'r') as data_file:
            file_iterator = iter(data_file)
            next(file_iterator)  # Skip the headers

            for line in file_iterator:
                tokens = line.split(',')
                sample_features = [float(t.strip()) for t in tokens]
                assert len(sample_features) == 4
                features.append(sample_features)

        label = label_map[file_id]

        if len(features) <= window:
            while len(features) < window:
                features.append([0, 0, 0, 0])

            sample_dict = {
                INPUTS: features,
                OUTPUT: label,
                ORIGINAL_ID: file_id
            }
            yield sample_dict
            continue

        indices = list(range(len(features)))
        for _ in range(reps):
            random_indices = np.sort(np.random.choice(indices, size=window, replace=False))
            feature_sample = [features[i] for i in random_indices]

            sample_dict = {
                INPUTS: feature_sample,
                OUTPUT: label,
                ORIGINAL_ID: file_id
            }

            yield sample_dict


def tokenize(data_folder: str, output_folder: str, window: int, reps: int, label_dict: Dict[int, int], partition_dict: Dict[int, int], chunk_size: int):

    make_dir(output_folder)
    writers = {
        1: DataWriter(os.path.join(output_folder, TRAIN), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size),
        2: DataWriter(os.path.join(output_folder, VALID), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size),
        3: DataWriter(os.path.join(output_folder, TEST), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size)
    }

    partition_counter = Counter()
    label_counters = [Counter(), Counter(), Counter()]

    data_generator = get_data_generator(data_folder, window, reps, label_dict)
    
    sample_id = 0
    for sample in data_generator:

        sample[SAMPLE_ID] = sample_id
        partition = partition_dict[sample[ORIGINAL_ID]]
        writers[partition].add(sample)
        sample_id += 1

        partition_counter[partition] += 1
        label_counters[partition - 1][sample[OUTPUT]] += 1

        if (sample_id + 1) % chunk_size == 0:
            print('Completed {0} samples'.format(sample_id + 1), end='\r')
    print()

    for writer in writers.values():
        writer.close()

    print(partition_counter)

    for label_counter, partition in zip(label_counters, [TRAIN, VALID, TEST]):
        total = sum(label_counter.values())
        for label, count in sorted(label_counter.items()):
            print('{0}, {1}: {2} ({3:.3f})'.format(partition, label, count, count / total))


if __name__ == '__main__':
    parser = ArgumentParser('Script to tokenize the HAR dataset.')
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--class-label-file', type=str, required=True)
    parser.add_argument('--partition-file', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--reps', type=int, required=True)
    parser.add_argument('--chunk-size', type=int, default=10000)
    args = parser.parse_args()

    assert args.window > 0, 'Must have a positive window'
    assert args.reps > 0, 'Must have a positive number of repetitions'

    assert os.path.exists(args.data_folder), 'The file {0} does not exist!'.format(args.data_folder)
    assert os.path.exists(args.class_label_file), 'The file {0} does not exist!'.format(args.class_label_file)
    assert os.path.exists(args.partition_file), 'The file {0} does not exist!'.format(args.partition_file)

    label_dict = load_class_labels(args.class_label_file)
    partition_dict = load_partition_map(args.partition_file)

    tokenize(data_folder=args.data_folder,
             output_folder=args.output_folder,
             window=args.window,
             reps=args.reps,
             label_dict=label_dict,
             partition_dict=partition_dict,
             chunk_size=args.chunk_size)
