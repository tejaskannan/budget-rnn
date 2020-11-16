import os.path
import re
import random
from argparse import ArgumentParser
from collections import Counter
from typing import Iterable, Dict, Any

from utils.constants import SAMPLE_ID, INPUTS, OUTPUT, TRAIN, VALID, TEST
from utils.data_writer import DataWriter
from utils.file_utils import make_dir


WINDOW_SIZE = 50
STRIDE = 15
CHUNK_SIZE = 5000
TRAIN_FRAC = 0.85
VALID_FRAC = 0.15

LABEL_MAP = {
    'cobblestone': 0,
    'dirt': 1,
    'flexible': 2
}

LINE_REGEX = re.compile(r'[,:]+')


def iterate_dataset(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, 'r') as fin:
        is_header = True
        sample_id = 0

        for line in fin:
            line = line.strip().lower()

            if line == '@data':
                is_header = False
            elif not is_header:
                tokens = LINE_REGEX.split(line)

                label = LABEL_MAP[tokens[-1]]
                features = list(map(float, tokens[:-1]))

                for i in range(0, len(features) - WINDOW_SIZE, STRIDE):
                    input_features = features[i:i+WINDOW_SIZE]

                    if len(input_features) == WINDOW_SIZE:
                        yield {
                            SAMPLE_ID: sample_id,
                            INPUTS: [[x] for x in input_features],  # Reshape to a 2D array
                            OUTPUT: label
                        }

                        sample_id += 1


def write_dataset(path: str, output_folder: str, series: str):
    # Create the data writers
    if series == TRAIN:
        writers = {
            TRAIN: DataWriter(os.path.join(output_folder, TRAIN), file_prefix='data', chunk_size=CHUNK_SIZE, file_suffix='jsonl.gz'),
            VALID: DataWriter(os.path.join(output_folder, VALID), file_prefix='data', chunk_size=CHUNK_SIZE, file_suffix='jsonl.gz')
        }

        label_counters = {
            TRAIN: Counter(),
            VALID: Counter()
        }
    else:
        writers = {
            TEST: DataWriter(os.path.join(output_folder, TEST), file_prefix='data', chunk_size=CHUNK_SIZE, file_suffix='jsonl.gz')
        }

        label_counters = {
            TEST: Counter()
        }

    # Iterate over the dataset
    for index, sample in enumerate(iterate_dataset(path=path)):

        # Select the partition
        if series == TRAIN:
            if random.random() < TRAIN_FRAC:
                partition = TRAIN
            else:
                partition = VALID
        else:
            partition = TEST

        writers[partition].add(sample)
        label_counters[partition][sample[OUTPUT]] += 1

        if (index + 1) % CHUNK_SIZE == 0:
            print('Completed {0} samples.'.format(index + 1), end='\r')
    print()

    # Close all data writers
    for writer in writers.values():
        writer.close()

    print(label_counters)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    args = parser.parse_args()

    # Set the random seed for reproducible results
    random.seed(42)

    # Create the output folder
    make_dir(args.output_folder)

    train_path = os.path.join(args.input_folder, 'AsphaltPavementType_TRAIN.ts')
    write_dataset(train_path, args.output_folder, series=TRAIN)

    test_path = os.path.join(args.input_folder, 'AsphaltPavementType_TEST.ts')
    write_dataset(test_path, args.output_folder, series=TEST)
