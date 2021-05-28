import os.path
import numpy as np
import random
from argparse import ArgumentParser
from collections import Counter

from utils.data_writer import DataWriter
from utils.file_utils import make_dir
from utils.constants import TRAIN, VALID, TEST, SAMPLE_ID, INPUTS, OUTPUT


WINDOW_SIZE = 20
STRIDE = 4
TRAIN_FRAC = 0.85
VALID_FRAC = 0.15
CHUNK_SIZE = 5000


def write_dataset(data: np.ndarray, output_folder: str, series: str):
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

    sample_id = 0
    for index, features in enumerate(data):
        label = int(features[0])
        input_features = features[1:].reshape(-1, 1).astype(float).tolist()

        # Get the data partition
        if series == TRAIN:
            if random.random() < TRAIN_FRAC:
                partition = TRAIN
            else:
                partition = VALID
        else:
            partition = TEST

        # Create the sample and add to corresponding data writer
        for i in range(0, len(input_features) - WINDOW_SIZE + 1, STRIDE):
            sample = {
                SAMPLE_ID: sample_id,
                OUTPUT: label,
                INPUTS: input_features[i:i+WINDOW_SIZE],
            }

            writers[partition].add(sample)
            label_counters[partition][label] += 1
            sample_id += 1

        if (index + 1) % CHUNK_SIZE == 0:
            print('Completed {0} sample.'.format(index + 1), end='\r')

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

    train_path = os.path.join(args.input_folder, 'MelbournePedestrian_TRAIN.txt')
    train_data = np.loadtxt(train_path)  # [T, D + 1] array. Element 0 is the label.

    make_dir(args.output_folder)
    write_dataset(train_data, output_folder=args.output_folder, series=TRAIN)

    test_path = os.path.join(args.input_folder, 'MelbournePedestrian_TEST.txt')
    test_data = np.loadtxt(test_path)
    write_dataset(test_data, output_folder=args.output_folder, series=TEST)

