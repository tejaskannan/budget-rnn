import os.path
from argparse import ArgumentParser
from collections import Counter, deque, defaultdict
from typing import Iterable, Dict, Any, List, Tuple, DefaultDict
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils.data_writer import DataWriter
from utils.constants import INPUTS, OUTPUT, SAMPLE_ID, TRAIN, VALID, TEST
from utils.file_utils import read_by_file_suffix, make_dir


LABEL_MAP = {
    'background': 0,
    'wine': 1,
    'banana': 2
}

SEQ_ID = 'sequence_id'


def create_partitions(metadata_path: str, train_frac: float, valid_frac: float) -> Dict[int, str]:
    label_id_map: DefaultDict[int, List[int]] = defaultdict(list)

    with open(metadata_path, 'r') as metadata_file:
        line_iterator = iter(metadata_file)
        next(line_iterator)  # Skip the headers

        for line in line_iterator:
            tokens = line.split()

            sample_id = int(tokens[0])
            label = LABEL_MAP[tokens[2]]
            label_id_map[label].append(sample_id)

    partitions: Dict[int, str] = dict()
    partition_counter: Counter = Counter()
    for label, sample_ids in label_id_map.items():
        sorted_ids = sorted(sample_ids)

        train_split_point = int(train_frac * len(sample_ids))
        valid_split_point = train_split_point + int(valid_frac * len(sample_ids))

        for index, sample_id in enumerate(sorted_ids):
            if index <= train_split_point:
                partitions[sample_id] = TRAIN
                partition_counter[TRAIN] += 1
            elif index <= valid_split_point:
                partitions[sample_id] = VALID
                partition_counter[VALID] += 1
            else:
                partitions[sample_id] = TEST
                partition_counter[TEST] += 1

    return partitions


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

            # The simulus is presented at time zero, so we skip the background samples beforehand
            if float(tokens[1]) < 0:
                continue

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
    partitions = create_partitions(metadata_file, train_frac=train_frac, valid_frac=valid_frac)

    # Create data writers
    make_dir(output_folder)
    writers = {
        TRAIN: DataWriter(os.path.join(output_folder, TRAIN), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size),
        VALID: DataWriter(os.path.join(output_folder, VALID), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size),
        TEST: DataWriter(os.path.join(output_folder, TEST), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size)
    }

    label_counter = Counter()
    data_window: deque = deque()

    stride_counter = stride
    sample_id = 0
    for sample in get_data_iterator(data_file, metadata_file):
        # Ensure that all samples come from the same input sequence
        while (len(data_window) > 0) and (data_window[0][SEQ_ID] != sample[SEQ_ID]):
            data_window.popleft()

        # Add in the new sample
        data_window.append(sample)
        
        # Remove excess data entries
        while len(data_window) > window:
            data_window.popleft()

        stride_counter += 1

        # Only write when the stride is fully reached
        if stride_counter >= stride and len(data_window) == window: 

            label = majority([elem[OUTPUT] for elem in data_window])
            features = [deepcopy(elem[INPUTS]) for elem in data_window]

            sample_dict = {
                SAMPLE_ID: sample_id,
                INPUTS: features,
                OUTPUT: label
            }

            seq_id = data_window[-1][SEQ_ID]
            partition = partitions[seq_id]

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
