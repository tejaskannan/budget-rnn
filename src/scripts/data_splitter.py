from argparse import ArgumentParser
import os
from hashlib import md5
from typing import Dict, List, Any
from collections import defaultdict, Counter

from utils.constants import TRAIN, VALID, TEST, SAMPLE_ID, DATA_FIELDS, SMALL_NUMBER, INDEX_FILE
from utils.file_utils import save_by_file_suffix, make_dir
from utils.data_writer import DataWriter, NpzDataWriter
from dataset.data_manager import get_data_manager


PARTITIONS = [TRAIN, VALID, TEST]
MODULUS = 2**16
CHUNK_SIZE = 10000
FILE_TYPES = ['jsonl.gz', 'pkl.gz', 'npz']


def compute_hash(sample: Dict[str, Any]) -> int:
    assert SAMPLE_ID in sample, f'All samples must have an id field named: {SAMPLE_ID}'
    hash_token = str(sample[SAMPLE_ID]).encode()
    return int(md5(hash_token).hexdigest(), 16)


def get_partition_index(sample: Dict[str, Any], fractions: List[float]) -> int:
    partition_index = compute_hash(sample) % MODULUS

    bound = 0
    for index, fraction in enumerate(fractions):
        bound += int(MODULUS * fraction)
        if partition_index < bound:
            return index

    # Default to TRAIN
    return 0

def split_dataset(input_folder: str, output_folder: str, fractions: List[float], file_prefix: str, chunk_size: int, file_type: str):
    assert len(fractions) == len(PARTITIONS), f'Must provide enough fractions to account for all partitions'
    assert file_type in FILE_TYPES, f'Invalid file type: {file_type}'

    # Make output folder if necessary
    make_dir(output_folder)

    # Create the data manager
    data_manager = get_data_manager(input_folder, SAMPLE_ID, DATA_FIELDS, extension=file_type)
    data_manager.load()
    data_iterator = data_manager.iterate(should_shuffle=False, batch_size=chunk_size)
    num_samples = data_manager.length

    # Get folders for each partition
    train_folder = os.path.join(output_folder, TRAIN)
    valid_folder = os.path.join(output_folder, VALID)
    test_folder = os.path.join(output_folder, TEST)

    # Track counts per partition
    partition_counters: Counter = Counter()

    # Create data writers
    if file_type == 'npz':
        partition_writers = {
            TRAIN: NpzDataWriter(train_folder, file_prefix=file_prefix, file_suffix=file_type, chunk_size=chunk_size, sample_id_name=SAMPLE_ID, data_fields=DATA_FIELDS, mode='w'),
            VALID: NpzDataWriter(valid_folder, file_prefix=file_prefix, file_suffix=file_type, chunk_size=chunk_size, sample_id_name=SAMPLE_ID, data_fields=DATA_FIELDS, mode='w'),
            TEST: NpzDataWriter(test_folder, file_prefix=file_prefix, file_suffix=file_type, chunk_size=chunk_size, sample_id_name=SAMPLE_ID, data_fields=DATA_FIELDS, mode='w')
        }
    else:
        partition_writers = {
            TRAIN: DataWriter(train_folder, file_prefix=file_prefix, file_suffix=file_type, chunk_size=chunk_size, mode='w'),
            VALID: DataWriter(valid_folder, file_prefix=file_prefix, file_suffix=file_type, chunk_size=chunk_size, mode='w'),
            TEST: DataWriter(test_folder, file_prefix=file_prefix, file_suffix=file_type, chunk_size=chunk_size, mode='w')
        }

    # Write to chunked files
    for index, sample in enumerate(data_iterator):
        partition_index = get_partition_index(sample, fractions)
        partition_folder = PARTITIONS[partition_index]
    
        partition_writers[partition_folder].add(sample)
        partition_counters[partition_folder] += 1

        if (index + 1) % chunk_size == 0:
            print(f'Completed {index + 1}/{num_samples} samples.', end='\r')
    print()

    # Flush any remaining data samples
    for writer in partition_writers.values():
        writer.flush()

    # Print out metrics and save metadata
    print('====== RESULTS ======')
    total = sum(partition_counters.values())
    metadata: Dict[str, Dict[str, float]] = dict()
    for series in PARTITIONS:
        count = partition_counters[series]
        frac = count / total
        metadata[series] = dict(count=count, frac=frac)

        print(f'{series.capitalize()}: {count} ({frac:.03f})')

    metadata_file = os.path.join(output_folder, 'metadata.json')
    save_by_file_suffix(metadata, metadata_file)


if __name__ == '__main__':
    parser = ArgumentParser('Utility script to split an dataset into folds.')
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--train-frac', type=float, required=True)
    parser.add_argument('--valid-frac', type=float, required=True)
    parser.add_argument('--test-frac', type=float)
    parser.add_argument('--file-prefix', type=str, default='data')
    parser.add_argument('--file-type', type=str, choices=['jsonl.gz', 'pkl.gz', 'npz'])
    parser.add_argument('--chunk-size', type=int, default=1000)
    args = parser.parse_args()

    assert os.path.exists(args.input_folder), f'The folder {args.input_folder} does not exist.'

    test_frac = args.test_frac if args.test_frac is not None else 1.0 - args.train_frac - args.valid_frac

    # Validate fractions
    fractions = [args.train_frac, args.valid_frac, args.test_frac]
    assert abs(sum(fractions) - 1.0) < SMALL_NUMBER, f'The fractions must add up to 1.0'
    for frac in fractions:
        assert frac >= 0, 'All fractions must be non-negative'

    split_dataset(input_folder=args.input_folder,
                  output_folder=args.output_folder,
                  fractions=fractions,
                  chunk_size=args.chunk_size,
                  file_prefix=args.file_prefix,
                  file_type=args.file_type)
