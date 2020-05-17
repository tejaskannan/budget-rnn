import numpy as np
import os.path
from argparse import ArgumentParser
from collections import defaultdict, Counter
from random import random
from typing import Iterable, Dict, Any, Tuple, List, DefaultDict

from utils.file_utils import iterate_files, read_by_file_suffix, make_dir, save_by_file_suffix
from utils.data_writer import DataWriter
from utils.constants import TIMESTAMP, INPUTS, OUTPUT, SAMPLE_ID
from utils.constants import TRAIN, VALID, TEST, SMALL_NUMBER


def get_data_generator(input_folder: str) -> Iterable[Dict[str, Any]]:
    for data_file in iterate_files(input_folder, pattern=r'.*jsonl.gz'):
        for sample in read_by_file_suffix(data_file):
            # Validate sample by skipping those with NoneType values
            if np.any(np.isnan(sample[INPUTS])) or np.any(sample[INPUTS] == None):
                continue

            yield sample


def assign_partitions(input_folder: str, train_frac: float, valid_frac: float) -> Dict[int, str]:
    data_generator = get_data_generator(input_folder)

    # Find the 'split' timestamps
    timestamps: List[int] = []
    timestamp_map: DefaultDict[str, List[int]] = defaultdict(list)
    for sample in data_generator:
        timestamps.append(sample[TIMESTAMP])
        timestamp_map[sample[TIMESTAMP]].append(sample[SAMPLE_ID])

    train_max = np.percentile(timestamps, train_frac * 100)
    valid_max = np.percentile(timestamps, (train_frac + valid_frac) * 100)

    partitions: Dict[int, str] = dict()
    for timestamp, sample_ids in timestamp_map.items():
        for sample_id in sample_ids:

            partition = TEST
            if timestamp < train_max:
                partition = TRAIN
            elif timestamp < valid_max:
                partition = VALID

            partitions[sample_id] = partition

    return partitions


def split_dataset(input_folder: str, output_folder: str, train_frac: float, valid_frac: float, test_frac: float, downsample_map: Dict[int, float], file_prefix: str, chunk_size: int, precision: int):
    
    print('Creating data partitions...')
    partitions = assign_partitions(input_folder, train_frac, valid_frac)
    print('Created partitions. Starting to write data...')

    # Create data writers
    make_dir(output_folder)
    writers = {
        TRAIN: DataWriter(os.path.join(output_folder, TRAIN), file_prefix=file_prefix, file_suffix='jsonl.gz', chunk_size=chunk_size),
        VALID: DataWriter(os.path.join(output_folder, VALID), file_prefix=file_prefix, file_suffix='jsonl.gz', chunk_size=chunk_size),
        TEST: DataWriter(os.path.join(output_folder, TEST), file_prefix=file_prefix, file_suffix='jsonl.gz', chunk_size=chunk_size)
    }

    data_generator = get_data_generator(input_folder)
    partition_counter: Counter = Counter()

    sample_id = 0
    for sample in data_generator:
        
        sample_partition = partitions[sample[SAMPLE_ID]]

        inputs = np.round(sample[INPUTS], decimals=precision).tolist()

        # Create concrete samples by sampling down the input sequences
        data_sample = {
            INPUTS: inputs,
            OUTPUT: sample[OUTPUT],
            SAMPLE_ID: sample_id,
            TIMESTAMP: sample[TIMESTAMP]
        }

        sample_id += 1

        # Only downsample the training dataset
        r = random()
        if r < downsample_map.get(sample[OUTPUT], 1.0) or sample_partition != TRAIN:
            writers[sample_partition].add(data_sample)
            partition_counter[sample_partition] += 1

        if (sample_id % chunk_size) == 0:
            print('Completed {0} samples.'.format(sample_id), end='\r')
    print()

    # Flush all remaining samples
    for writer in writers.values():
        writer.flush()

    total = sum(partition_counter.values())
    for partition, count in sorted(partition_counter.items()):
        print('{0}: {1} ({2:.3f})'.format(partition, count, count / total))

    metadata = {
        'train_count': partition_counter[TRAIN],
        'train_frac': partition_counter[TRAIN] / total,
        'valid_count': partition_counter[VALID],
        'valid_frac': partition_counter[VALID] / total,
        'test_count': partition_counter[TEST],
        'test_frac': partition_counter[TEST] / total,
        'total': total
    }
    metadata_file = os.path.join(output_folder, 'metadata.json')
    save_by_file_suffix(metadata, metadata_file)


if __name__ == '__main__':
    parser = ArgumentParser('Script to create dataset folds based on timestamps')
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--train-frac', type=float, required=True)
    parser.add_argument('--valid-frac', type=float, required=True)
    parser.add_argument('--test-frac', type=float)
    parser.add_argument('--downsample-fracs', type=float, nargs='*')
    parser.add_argument('--downsample-labels', type=int, nargs='*')
    parser.add_argument('--file-prefix', type=str, default='data')
    parser.add_argument('--chunk-size', type=int, default=5000)
    parser.add_argument('--precision', type=int, default=4)
    args = parser.parse_args()

    # Validate fractions and downsample parameters
    test_frac = args.test_frac if args.test_frac is not None else 1.0 - args.train_frac - args.valid_frac
    assert abs(args.train_frac + args.valid_frac + test_frac - 1.0) < SMALL_NUMBER, 'The fractions must add up to one'
    assert args.train_frac > 0, 'The training fraction must be positive.'
    assert args.valid_frac >= 0, 'The validation fraction must be non-negative.'
    assert test_frac >= 0, 'The testing fraction must be non-negative.'
    assert (args.downsample_fracs is None and args.downsample_labels is None) or len(args.downsample_fracs) == len(args.downsample_labels), 'Misaligned downsample parameters.'
    assert args.precision > 0, 'Must have a positive precision'

    downsample_map: Dict[int, float] = dict()
    if args.downsample_fracs is not None:
        for label, frac in zip(args.downsample_labels, args.downsample_fracs):
            downsample_map[label] = frac

    split_dataset(input_folder=args.input_folder,
                  output_folder=args.output_folder,
                  train_frac=args.train_frac,
                  valid_frac=args.valid_frac,
                  test_frac=test_frac,
                  downsample_map=downsample_map,
                  file_prefix=args.file_prefix,
                  chunk_size=args.chunk_size,
                  precision=args.precision)
