import numpy as np
import os
from hashlib import md5
from argparse import ArgumentParser
from typing import Dict, List, Any

from utils.constants import TRAIN, VALID, TEST, SAMPLE_ID, DATA_FIELDS, SMALL_NUMBER, DATA_FIELD_FORMAT, INDEX_FILE
from utils.npz_data_manager import NpzDataManager
from utils.file_utils import save_by_file_suffix


PARTITIONS = [TRAIN, VALID, TEST]
MODULUS = 2**16


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


def split_dataset(input_folder: str, output_folder: str, fractions: List[float], file_prefix: str, chunk_size: int):
    assert len(fractions) == len(PARTITIONS), f'Must provide enough fractions to account for all partitions'

    # Make output folders if necessary
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    # Make partition folders if necessary
    for partition_folder in PARTITIONS:
        partition_path = os.path.join(output_folder, partition_folder)
        if not os.path.exists(partition_path):
            os.mkdir(partition_path)

    # Load data and create data iterator
    data_manager = NpzDataManager(input_folder, SAMPLE_ID, DATA_FIELDS)
    data_manager.load()
    data_iterator = data_manager.iterate(should_shuffle=False, batch_size=chunk_size)
    num_samples = data_manager.length

    # Hold partitions before flushing to output files
    partition_dicts: List[Dict[str, Any]] = [dict() for _ in PARTITIONS]
    partition_file_indexes = [0 for _ in PARTITIONS]
    partition_counts = [0 for _ in PARTITIONS]
    partition_indexes: List[Dict[int, int]] = [dict() for _ in PARTITIONS]

    normalize_array = np.array([[540, 960, 540, 960, 540, 960, 540, 960]])

    for index, sample in enumerate(data_iterator):
        partition_index = get_partition_index(sample, fractions)
        partition_folder = PARTITIONS[partition_index]
        partition_dict = partition_dicts[partition_index]

        # Add this sample by giving each field a unique name
        sample_dict: Dict[str, Any] = dict()
        sample_id = sample[SAMPLE_ID]

        for field in DATA_FIELDS:
            field_name = DATA_FIELD_FORMAT.format(field, sample_id)
            sample_dict[field_name] = sample[field]

        partition_dict.update(**sample_dict)
        partition_indexes[partition_index][sample_id] = partition_file_indexes[partition_index]
        partition_counts[partition_index] += 1

        partition_samples = len(partition_dict) / len(DATA_FIELDS)
        if partition_samples >= chunk_size:
            file_index = partition_file_indexes[partition_index]
            output_file = os.path.join(output_folder, partition_folder, f'{file_prefix}{file_index:03}.npz')
            np.savez_compressed(output_file, **partition_dict)
            partition_dicts[partition_index] = dict()
            partition_file_indexes[partition_index] += 1

            print(f'Completed {index + 1}/{num_samples} samples.', end='\r')

    # Save remaining elements
    for partition_index, partition_dict in enumerate(partition_dicts):
            # Save the partition index
            partition_folder = PARTITIONS[partition_index]
            index_file = os.path.join(output_folder, partition_folder, INDEX_FILE)
            save_by_file_suffix(partition_indexes[partition_index], index_file)

            # Skip empty dictionaries
            if len(partition_dict) == 0:
                continue

            file_index = partition_file_indexes[partition_index]
            output_file = os.path.join(output_folder, partition_folder, f'{file_prefix}{file_index:03}.npz')
            np.savez_compressed(output_file, **partition_dict)
            partition_dicts[partition_index] = dict()
            partition_file_indexes[partition_index] += 1

    # Save meta-data
    total = float(sum(partition_counts))
    metadata = {
        'train_count': partition_counts[0],
        'valid_count': partition_counts[1],
        'test_count': partition_counts[2],
        'train_frac': partition_counts[0] / total,
        'valid_frac': partition_counts[1] / total,
        'test_frac': partition_counts[2] / total,
        'total': total
    }
    save_by_file_suffix([metadata], os.path.join(output_folder, 'metadata.jsonl.gz'))

    print()
    for key, value in sorted(metadata.items()):
        print(f'{key}: {value}')


if __name__ == '__main__':
    parser = ArgumentParser('Utility script to split a dataset into folds.')
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--train-frac', type=float, required=True)
    parser.add_argument('--valid-frac', type=float, required=True)
    parser.add_argument('--test-frac', type=float)
    parser.add_argument('--file-prefix', type=str, default='data')
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
                  file_prefix=args.file_prefix)
