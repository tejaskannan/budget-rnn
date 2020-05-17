import os.path
from argparse import ArgumentParser
from collections import Counter
from typing import Dict, Iterable, Any

from utils.data_writer import DataWriter
from utils.file_utils import make_dir
from utils.constants import INPUTS, OUTPUT, SAMPLE_ID, TRAIN, VALID, TEST


def get_data_generator(input_path: str) -> Iterable[Dict[str, Any]]:
    with open(input_path, 'r') as input_file:
        for sample_id, line in enumerate(input_file):
            tokens = line.split(',')

            features = list(map(float, tokens[:-1]))
            label = int(tokens[-1])

            yield {
                SAMPLE_ID: sample_id,
                INPUTS: features,
                OUTPUT: label
            }

def get_num_samples(input_path: str) -> int:
    with open(input_path, 'r') as input_file:
        count = 0
        for _ in input_file:
            count += 1
    return count


def tokenize(input_file: str, output_folder: str, train_frac: float, valid_frac: float, chunk_size: int):
    # Determine split points
    num_samples = get_num_samples(input_file)
    train_point = int(train_frac * num_samples)
    valid_point = train_point + int(valid_frac * num_samples)

    data_generator = get_data_generator(input_file)

    make_dir(output_folder)
    writers = {
        TRAIN: DataWriter(os.path.join(output_folder, TRAIN), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size),
        VALID: DataWriter(os.path.join(output_folder, VALID), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size),
        TEST: DataWriter(os.path.join(output_folder, TEST), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size)
    }

    data_counters = {
        TRAIN: Counter(),
        VALID: Counter(),
        TEST: Counter()
    }

    for index, sample in enumerate(data_generator):
        if index < train_point:
            writers[TRAIN].add(sample)
            data_counters[TRAIN][sample[OUTPUT]] += 1
        elif index < valid_point:
            writers[VALID].add(sample)
            data_counters[VALID][sample[OUTPUT]] += 1
        else:
            writers[TEST].add(sample)
            data_counters[TEST][sample[OUTPUT]] += 1

    for writer in writers.values():
        writer.close()

    print(data_counters)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--train-frac', type=float, required=True)
    parser.add_argument('--valid-frac', type=float, required=True)
    parser.add_argument('--chunk-size', type=int, default=5000)
    args = parser.parse_args()

    tokenize(input_file=args.input_file,
             output_folder=args.output_folder,
             train_frac=args.train_frac,
             valid_frac=args.valid_frac,
             chunk_size=args.chunk_size)
