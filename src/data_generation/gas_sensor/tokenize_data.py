import os.path
from random import random
from argparse import ArgumentParser
from collections import Counter, deque, defaultdict
from typing import Iterable, Dict, Any, List, Tuple, DefaultDict
from hashlib import md5
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils.data_writer import DataWriter
from utils.constants import INPUTS, OUTPUT, SAMPLE_ID, TRAIN, VALID, TEST, TIMESTAMP
from utils.file_utils import read_by_file_suffix, make_dir, save_by_file_suffix


LABEL_MAP = {
    'background': 0,
    'wine': 1,
    'banana': 2
}

SEQ_ID = 'sequence_id'
MODULUS = 2**16


def compute_hash(timestamp: str) -> int:
    hash_token = timestamp.encode()
    return int(md5(hash_token).hexdigest(), 16)


def get_partition(timestamp: str, train_frac: float, valid_frac: float) -> str:
    partition_index = compute_hash(timestamp) % MODULUS

    train_bound = int(MODULUS * train_frac)
    valid_bound = int(MODULUS * valid_frac) + train_bound

    if partition_index < train_bound:
        return TRAIN
    if partition_index < valid_bound:
        return VALID

    return TEST


def create_partitions(metadata_path: str, train_frac: float, valid_frac: float, data_map: Dict[int, List[List[float]]]) -> Dict[int, str]:
    label_id_map: DefaultDict[int, List[int]] = defaultdict(list)

    with open(metadata_path, 'r') as metadata_file:
        line_iterator = iter(metadata_file)
        next(line_iterator)  # Skip the headers

        for line in line_iterator:
            tokens = line.split()

            sample_id = int(tokens[0])
            date = tokens[1]
            label = LABEL_MAP[tokens[2]]
            label_id_map[label].append((sample_id, date))

    partitions: Dict[int, str] = dict()
    
    partition_counter: Dict[str, Counter] = {
        TRAIN: Counter(),
        VALID: Counter(),
        TEST: Counter()
    }

    for label, sample_ids in label_id_map.items():

        for sample_id, timestamp in sample_ids:
            sample_len = len(data_map[sample_id])

            partition = get_partition(timestamp, train_frac, valid_frac)
            partitions[sample_id] = partition
            partition_counter[partition][label] += sample_len

    print(partition_counter)

    return partitions


def load_data(data_path: str) -> DefaultDict[int, List[Any]]:
    data_map: DefaultDict[int, List[Any]] = defaultdict(list)

    with open(data_path, 'r') as data_file:
        data_iterator = iter(data_file)
        next(data_iterator)  # Skip the headers

        for line in data_iterator:
            tokens = line.split()

            seq_id = int(tokens[0])
            timestamp = float(tokens[1])

            # Only include positive times because the stimulus is introduced
            # at time zero.
            if timestamp >= 0:
                sample_dict = {
                    TIMESTAMP: timestamp,
                    INPUTS: list(map(float, tokens[2:]))
                }
                data_map[seq_id].append(sample_dict)

    min_seq_length = min((len(values) for values in data_map.values()))
    max_seq_length = max((len(values) for values in data_map.values()))

    return data_map


def create_label_map(metadata_file: str) -> Dict[int, int]:
    id_label_map: Dict[int, int] = dict()
    with open(metadata_file, 'r') as metadata_file:
        for i, line in enumerate(metadata_file):
            # Skip header
            if i == 0:
                continue

            tokens = line.split()
            id_label_map[int(tokens[0])] = LABEL_MAP[tokens[2]]

    return id_label_map


def get_count_per_sequence(data_path: str) -> Counter:

    seq_counter: Counter = Counter()

    with open(data_path, 'r') as data_file:
        data_file_iterator = iter(data_file)
        next(data_file_iterator)  # Skip the headers

        for sample in data_file_iterator:
            tokens = sample.split()

            if float(tokens[1]) > 0:
                seq_counter[int(tokens[0])] += 1

    return seq_counter


def get_data_iterator(data_file: str, metadata_file: str) -> Iterable[Dict[str, Any]]:
    id_label_map: Dict[int, int] = dict()
    with open(metadata_file, 'r') as metadata_file:
        for i, line in enumerate(metadata_file):
            # Skip header
            if i == 0:
                continue

            tokens = line.split()
            id_label_map[tokens[0]] = LABEL_MAP[tokens[2]]

    with open(data_file, 'r') as features_file:
        for i, features in enumerate(features_file):
            if i == 0:
                continue

            tokens = [t.strip() for t in features.split(' ') if len(t.strip()) > 0]

            # The stimulus is presented at time zero, so we skip the background samples beforehand
           # if float(tokens[1]) < 0:
           #     continue

            try:
                feature_values = [float(val) for val in tokens[2:]]
                label = id_label_map[tokens[0]]
                sample = {
                    INPUTS: feature_values,
                    OUTPUT: int(label),
                    SEQ_ID: int(tokens[0])
                }

                yield sample
            except ValueError:
                raise


def majority(elements: List[int]) -> int:
    elem_counter: Counter = Counter()
    for x in elements:
        elem_counter[x] += 1
    return elem_counter.most_common(1)[0][0]


def tokenize_data(data_file: str, metadata_file: str, output_folder: str, window: int, stride: int, chunk_size: int, train_frac: float, valid_frac: float):
    print('Loading dataset...')
    data_map = load_data(data_file)
    seq_label_map = create_label_map(metadata_file)
    print('Finished loading data. Starting to tokenize.')

    partitions = create_partitions(metadata_file, train_frac=train_frac, valid_frac=valid_frac, data_map=data_map)

    # Create data writers
    make_dir(output_folder)
    writers = {
        TRAIN: DataWriter(os.path.join(output_folder, TRAIN), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size),
        VALID: DataWriter(os.path.join(output_folder, VALID), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size),
        TEST: DataWriter(os.path.join(output_folder, TEST), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size)
    }

    label_counter = Counter()

    sample_id = 0
    for seq_id, samples in data_map.items():
        label = seq_label_map[seq_id]
        partition = partitions[seq_id]

        # Choose stride to even out the partitions
        stride_counter = stride

        data_window: deque = deque()
        for sample in samples:
        
            # Add in the new sample
            data_window.append(sample)
        
            # Remove excess data entries
            while len(data_window) > window:
                data_window.popleft()

            stride_counter += 1

            # Only write when the stride is fully reached
            if stride_counter >= stride and len(data_window) == window: 

                features = [deepcopy(elem[INPUTS]) for elem in data_window]

                sample_dict = {
                    SAMPLE_ID: sample_id,
                    INPUTS: features,
                    OUTPUT: label
                }

                writers[partition].add(sample_dict)
                stride_counter = 0
                sample_id += 1
                label_counter[label] += 1

            if (sample_id + 1) % chunk_size == 0:
                print('Completed {0} samples.'.format(sample_id + 1), end='\r')

    print()
    print(label_counter)

    for writer in writers.values():
        writer.close()

    # Save the partition mappings for future use
    partition_file = os.path.join(output_folder, 'partitions.json')
    save_by_file_suffix(partitions, partition_file)


if __name__ == '__main__':
    parser = ArgumentParser('Script to tokenize Eye sensor dataset.')
    parser.add_argument('--data-file', type=str, required=True)
    parser.add_argument('--metadata-file', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--train-frac', type=float, required=True)
    parser.add_argument('--valid-frac', type=float, required=True)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--stride', type=int, required=True)
    parser.add_argument('--chunk-size', type=int, required=True)
    args = parser.parse_args()

    tokenize_data(data_file=args.data_file,
                  metadata_file=args.metadata_file,
                  output_folder=args.output_folder,
                  window=args.window,
                  stride=args.stride,
                  chunk_size=args.chunk_size,
                  train_frac=args.train_frac,
                  valid_frac=args.valid_frac)
