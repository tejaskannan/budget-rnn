from argparse import ArgumentParser
from dpu_utils.utils import RichPath, ChunkWriter
from datetime import datetime
from hashlib import md5
from enum import Enum, auto
from collections import Counter
from typing import Dict, List, Any
from itertools import chain


TRAIN_DIR = 'train'
VALID_DIR = 'valid'
TEST_DIR = 'test'
MODULUS = 2**16
DATA_FILE_NAME = 'data'
MAX_CHUNK_SIZE = 10000


class DataPartition(Enum):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


PARTITIONS = {
        DataPartition.TRAIN: TRAIN_DIR,
        DataPartition.VALID: VALID_DIR,
        DataPartition.TEST: TEST_DIR
}


def compute_hash(sample: Dict[str, Any], hash_fields: List[str]) -> int:
    tokens: List[str] = []
    for field in hash_fields:
        field_val = sample[field]
        if isinstance(field_val, list):
            tokens.extend(field_val)
        else:
            tokens.append(field_val)

    hash_token = '-'.join(tokens).encode()
    return int(md5(hash_token).hexdigest(), 16)


def get_partition(sample: Dict[str, Any], fracs: Dict[DataPartition, float], hash_fields: List[str]) -> DataPartition:
    partition_index = compute_hash(sample, hash_fields) % MODULUS

    bound = 0
    for partition, frac in fracs.items():
        bound += int(MODULUS * frac)
        if partition_index < bound:
            return partition
    return DataPartition.TRAIN


def get_partition_sequential(sample: Dict[str, Any], split_field: str, split_points: Dict[DataPartition, Any]) -> DataPartition:
    current_partition, current_point = None, None
    
    try:
        split_value = datetime.strptime(sample[split_field], '%Y-%m-%dT%H:%M:%S')
    except ValueError:
        split_value = sample[split_field]

    for partition, split_point in split_points.items():
        if split_value <= split_point and (current_point is None or split_point < current_point):
            current_point = split_point
            current_partition = partition

    if current_partition is None:
        return DataPartition.TRAIN

    return current_partition


def get_split_points(input_dir: RichPath, fracs: Dict[DataPartition, float], split_field: str) -> Dict[DataPartition, Any]:
    data_files = sorted(input_dir.iterate_filtered_files_in_dir('data*.jsonl.gz'))
    data_samples = chain(*(data_file.read_by_file_suffix() for data_file in data_files))

    split_field_values: List[Any] = []
    for sample in data_samples:
        try:
            date_value = datetime.strptime(sample[split_field], '%Y-%m-%dT%H:%M:%S')
            split_field_values.append(date_value)
        except ValueError:
            split_field_values.append(samples[split_field])

    split_field_values = list(sorted(split_field_values))

    split_points: Dict[DataPartition, Any] = dict()
    offset = 0
    for partition, frac in fracs.items():
        step = int(frac * len(split_field_values))
        split_index = min(len(split_field_values) - 1, step + offset)
        split_points[partition] = split_field_values[split_index]
        offset += step

    return split_points


def split_dataset(input_dir: RichPath, output_dir: RichPath, fracs: Dict[DataPartition, float],
                  hash_fields: List[str], mode: str):
    output_dir.make_as_dir()

    train_dir = output_dir.join(TRAIN_DIR)
    train_dir.make_as_dir()

    valid_dir = output_dir.join(VALID_DIR)
    valid_dir.make_as_dir()

    test_dir = output_dir.join(TEST_DIR)
    test_dir.make_as_dir()

    if mode == 'sequential':
        split_points = get_split_points(input_dir, fracs, hash_fields[0])

    data_files = sorted(input_dir.iterate_filtered_files_in_dir('data*.jsonl.gz'))
    data_samples = chain(*(data_file.read_by_file_suffix() for data_file in data_files))
    
    data_counters = Counter()
    with ChunkWriter(train_dir, DATA_FILE_NAME, MAX_CHUNK_SIZE, file_suffix='.jsonl.gz', parallel_writers=0) as train_writer,\
        ChunkWriter(valid_dir, DATA_FILE_NAME, MAX_CHUNK_SIZE, file_suffix='.jsonl.gz', parallel_writers=0) as valid_writer,\
        ChunkWriter(test_dir, DATA_FILE_NAME, MAX_CHUNK_SIZE, file_suffix='.jsonl.gz', parallel_writers=0) as test_writer:

        writers = {
                DataPartition.TRAIN: train_writer,
                DataPartition.VALID: valid_writer,
                DataPartition.TEST: test_writer
        }

        total = 0
        for sample in data_samples:
            if mode == 'random':
                partition = get_partition(sample, fracs, hash_fields)
            else:
                partition = get_partition_sequential(sample, hash_fields[0], split_points)
            
            writers[partition].add(sample)
            data_counters[partition] += 1
            total += 1

            if total % MAX_CHUNK_SIZE:
                print(f'Completed {total} samples.', end='\r')


    train_frac = data_counters[DataPartition.TRAIN] / float(total)
    valid_frac = data_counters[DataPartition.VALID] / float(total)
    test_frac = data_counters[DataPartition.TEST] / float(total)

    # Back up the metadata_file
    metadata_file = input_dir.join('metadata.jsonl.gz')
    if metadata_file.exists():
        metadata_list = list(metadata_file.read_by_file_suffix())
        metadata = metadata_list[0]
        metadata['train_frac'] = fracs[DataPartition.TRAIN]
        metadata['valid_frac'] = fracs[DataPartition.VALID]
        metadata['test_frac'] = fracs[DataPartition.TEST]

        output_dir.join('metadata.jsonl.gz').save_as_compressed_file([metadata])

    print()
    print(f'Total Number of Samples: {total}')
    print(f'Fraction in Training Dataset ({data_counters[DataPartition.TRAIN]}): {train_frac: .4f}')
    print(f'Fraction in Validation Dataset ({data_counters[DataPartition.VALID]}): {valid_frac: .4f}')
    print(f'Fraction in Testing Dataset ({data_counters[DataPartition.TEST]}): {test_frac: .4f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--hash-fields', type=str, nargs='+')
    parser.add_argument('--train-frac', type=float, required=True)
    parser.add_argument('--valid-frac', type=float, required=True)
    parser.add_argument('--mode', type=str, choices=['random', 'sequential'])
    args = parser.parse_args()

    if args.train_frac + args.valid_frac >= 1:
        raise ValueError('The training and validation fractions must sum to less than one.')

    fractions = {
            DataPartition.TRAIN: args.train_frac,
            DataPartition.VALID: args.valid_frac,
            DataPartition.TEST: (1.0 - args.train_frac - args.valid_frac)
    }

    split_dataset(input_dir=RichPath.create(args.input_folder),
                  output_dir=RichPath.create(args.output_folder),
                  fracs=fractions,
                  hash_fields=args.hash_fields,
                  mode=args.mode)
