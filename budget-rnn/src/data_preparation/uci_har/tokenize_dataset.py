import os
import re
import numpy as np
from argparse import ArgumentParser
from collections import namedtuple, Counter
from typing import Dict, Iterable, Any

from utils.constants import TRAIN, VALID, TEST, INPUTS, OUTPUT, SAMPLE_ID
from utils.data_writer import DataWriter
from utils.file_utils import make_dir


LabelKey = namedtuple('LabelKey', ['user_id', 'exp_id', 'begin', 'end'])
DataKey = namedtuple('DataKey', ['user_id', 'exp_id'])
DataArrays = namedtuple('DataArrays', ['acc', 'gyr'])

WINDOW_SIZE = 50
STRIDE = 20
LABELS_FILE = 'labels.txt'

# Remaining are for training
VALID_USER_IDS = set([3, 5, 11, 14, 19, 29])
TEST_USER_IDS = set([2, 4, 9, 10, 12, 13, 18, 20, 24])


def load_data(input_folder: str, arrays: DataArrays, begin: int, end: int) -> np.ndarray:
    for i in range(begin, end - WINDOW_SIZE, STRIDE):
        acc_features = arrays.acc[i:i+WINDOW_SIZE, :]
        gyr_features = arrays.gyr[i:i+WINDOW_SIZE, :]

        data_features = np.concatenate([acc_features, gyr_features], axis=-1)

        if data_features.shape[0] == WINDOW_SIZE:
            yield data_features


def get_partition(user_id: int) -> str:
    if user_id in VALID_USER_IDS:
        return VALID
    elif user_id in TEST_USER_IDS:
        return TEST
    return TRAIN


def load_data_files(labels_dict: Dict[LabelKey, int], input_folder: str) -> Dict[DataKey, DataArrays]:
    result: Dict[DataKey, DataArrays] = dict()
    for label_key in labels_dict.keys():
        key = DataKey(user_id=label_key.user_id, exp_id=label_key.exp_id)

        # Don't load files twice
        if key in result:
            continue

        # Load the input files
        acc = np.loadtxt(os.path.join(input_folder, 'acc_exp{0:02d}_user{1:02d}.txt'.format(label_key.exp_id, label_key.user_id)))
        gyr = np.loadtxt(os.path.join(input_folder, 'gyro_exp{0:02d}_user{1:02d}.txt'.format(label_key.exp_id, label_key.user_id)))

        key = DataKey(user_id=label_key.user_id, exp_id=label_key.exp_id)
        arrays = DataArrays(acc=acc, gyr=gyr)

        result[key] = arrays

    return result

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--chunk-size', type=int, default=10000)
    args = parser.parse_args()

    # Read the labels
    labels = np.loadtxt(os.path.join(args.input_folder, LABELS_FILE))

    labels_dict: Dict[LabelKey, int] = dict()
    for entry in labels:
        key = LabelKey(user_id=int(entry[1]), exp_id=int(entry[0]), begin=int(entry[3]), end=int(entry[4]))
        labels_dict[key] = int(entry[2])

    # Create the output data writers
    make_dir(args.output_folder)
    writers = {
        TRAIN: DataWriter(os.path.join(args.output_folder, TRAIN), file_prefix='data', file_suffix='jsonl.gz', chunk_size=args.chunk_size),
        VALID: DataWriter(os.path.join(args.output_folder, VALID), file_prefix='data', file_suffix='jsonl.gz', chunk_size=args.chunk_size),
        TEST: DataWriter(os.path.join(args.output_folder, TEST), file_prefix='data', file_suffix='jsonl.gz', chunk_size=args.chunk_size),
    }

    # Initialize counters
    counters = {
        TRAIN: Counter(),
        VALID: Counter(),
        TEST: Counter()
    }

    # We load all data files first to prevent redundant loading
    print('Loading Input files...')
    data_files = load_data_files(labels_dict, args.input_folder)

    print('Writing data...')
    sample_id = 0
    for key, label in labels_dict.items():

        partition = get_partition(user_id=key.user_id)
        writer = writers[partition]
        counter = counters[partition]

        data_arrays = data_files[DataKey(user_id=key.user_id, exp_id=key.exp_id)]

        for data_features in load_data(args.input_folder, arrays=data_arrays, begin=key.begin, end=key.end):
       
            sample = {
                INPUTS: data_features.astype(float).tolist(),
                OUTPUT: label,
                SAMPLE_ID: sample_id
            }
            writer.add(sample)

            counter[label] += 1

            sample_id += 1

            if sample_id % args.chunk_size == 0:
                print('Completed {0} samples'.format(sample_id), end='\r')

    print()

    print('====== Label distribution =====')
    print(counters)

    # Close all writers
    for writer in writers.values():
        writer.close()
