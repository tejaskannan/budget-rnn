import os
import random
from argparse import ArgumentParser
from collections import Counter
from typing import Iterable, Dict, Any, List

from utils.data_writer import DataWriter
from utils.file_utils import make_dir
from utils.constants import TRAIN, VALID, TEST, SAMPLE_ID, INPUTS, OUTPUT


def read_dataset(input_path: str) -> Iterable[Dict[str, Any]]:
    with open(input_path, 'r') as input_file:
        is_header = True
        sample_id = 0

        for line in input_file:
            line = line.strip().lower()
            if line == '@data':
                is_header = False
            elif not is_header:
                tokens = line.split(':')

                xs = list(map(float, tokens[0].split(',')))
                ys = list(map(float, tokens[1].split(',')))

                label = int(tokens[-1])

                features = [[x, y] for x, y in zip(xs, ys)]

                yield {
                    SAMPLE_ID: sample_id,
                    INPUTS: features,
                    OUTPUT: label
                }
                sample_id += 1


def get_partition(partitions: List[str], fractions: List[float]) -> str:
    r = random.random()

    frac_sum = 0.0
    for frac, partition in zip(fractions, partitions):
        frac_sum += frac
        if r < frac_sum:
            return partition

    return partitions[-1]


def write_dataset(dataset: List[Dict[str, Any]], partitions: List[str], fractions: List[float], output_folder: str):
    # Initialize writers and counters
    writers: Dict[str, DataWriter] = dict()
    label_counters: Dict[str, Counter] = dict()
    for partition in partitions:
        writer = DataWriter(os.path.join(output_folder, partition), chunk_size=5000, file_prefix='data', file_suffix='jsonl.gz')
        writers[partition] = writer

        label_counters[partition] = Counter()

    # Write all samples
    for sample in dataset:
        partition = get_partition(partitions, fractions)
        writers[partition].add(sample)
        label_counters[partition][sample[OUTPUT]] += 1

    # Close all writers
    for writer in writers.values():
        writer.close()

    print(label_counters)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    args = parser.parse_args()

    train_file = os.path.join(args.input_folder, 'PenDigits_TRAIN.ts')
    test_file = os.path.join(args.input_folder, 'PenDigits_TEST.ts')

    train_dataset = list(read_dataset(train_file))
    test_dataset = list(read_dataset(test_file))

    # Set random seed for reproducible results
    random.seed(42)

    make_dir(args.output_folder)
    write_dataset(train_dataset, partitions=[TRAIN, VALID], fractions=[0.8, 0.2], output_folder=args.output_folder)
    write_dataset(test_dataset, partitions=[TEST], fractions=[1.0], output_folder=args.output_folder)
