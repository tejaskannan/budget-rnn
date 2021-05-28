import os
import numpy as np
from argparse import ArgumentParser
from collections import Counter
from typing import Iterable, Dict, Any, Tuple

from utils.constants import TRAIN, VALID, TEST, SAMPLE_ID, INPUTS, OUTPUT
from utils.file_utils import make_dir
from utils.data_writer import DataWriter


WINDOW = 50
STRIDE = 25
DOWNSAMPLE_SKIP = 3


def get_partition(subject_id: int) -> str:
    if subject_id <= 10:
        return TEST
    elif subject_id <= 15:
        return VALID
    else:
        return TRAIN


def data_generator(input_folder: str) -> Iterable[Tuple[Dict[str, Any], str]]:
    sample_id = 0
    for subject_id in sorted(os.listdir(args.input_folder)):
        folder = os.path.join(args.input_folder, subject_id)
        if not os.path.isdir(folder):
            continue

        for data_file in os.listdir(folder):
            try:
                dataset = np.loadtxt(os.path.join(folder, data_file),
                                     skiprows=1,
                                     dtype=str)
                downsampled_dataset = dataset[::DOWNSAMPLE_SKIP]

                for start in range(0, downsampled_dataset.shape[0] - WINDOW + 1, STRIDE):
                    end = start + WINDOW
                    data_chunk = downsampled_dataset[start:end].astype(float)

                    # Element 0 is the timestamp, and the final element is the class label
                    input_features = data_chunk[:, 1:-1]
                    labels = data_chunk[:, -1]

                    if all((label != 0 for label in labels)) and len(input_features) == WINDOW:
                        sample_dict = {
                            SAMPLE_ID: sample_id,
                            OUTPUT: labels[-1],
                            INPUTS: input_features.astype(float).tolist()
                        }
                        yield sample_dict, get_partition(int(subject_id))

                        sample_id += 1

            except ValueError as ex:
                print(data_file)
                print(ex)


def tokenize_dataset(input_folder: str, output_folder: str, chunk_size: int):
    make_dir(output_folder)
    data_writers = {
        TRAIN: DataWriter(os.path.join(output_folder, TRAIN), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size),
        VALID: DataWriter(os.path.join(output_folder, VALID), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size),
        TEST: DataWriter(os.path.join(output_folder, TEST), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size)
    }

    partition_counters = {
        TRAIN: Counter(),
        VALID: Counter(),
        TEST: Counter()
    }

    for i, (sample, partition) in enumerate(data_generator(input_folder)):
        data_writers[partition].add(sample)
        partition_counters[partition][sample[OUTPUT]] += 1

        if (i + 1) % chunk_size == 0:
            print('Wrote {0} samples.'.format(i+1), end='\r')
    print()

    for writer in data_writers.values():
        writer.close()

    print(partition_counters)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--chunk-size', type=int, default=5000)
    args = parser.parse_args()

    tokenize_dataset(args.input_folder, args.output_folder, args.chunk_size)
