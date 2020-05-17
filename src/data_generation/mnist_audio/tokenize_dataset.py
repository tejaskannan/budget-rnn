import os.path
import re
import scipy.io.wavfile as wav
import numpy as np
from argparse import ArgumentParser
from collections import Counter
from typing import List, Iterable, Tuple

from utils.data_writer import DataWriter
from utils.file_utils import iterate_files, make_dir
from utils.constants import TRAIN, VALID, TEST, SAMPLE_ID, INPUTS, OUTPUT, SMALL_NUMBER


FILE_REGEX = re.compile(r'([0-9])_[a-zA-Z]+_([0-9]+)\.wav')


def get_label_and_partition(file_path: str) -> Tuple[int, str]:
    _, file_name = os.path.split(file_path)

    # Extract the label and file index
    match = FILE_REGEX.match(file_name)
    label = int(match.group(1))
    index = int(match.group(2))

    if index < 5:
        return label, TEST
    elif index < 10:
        return label, VALID
    else:
        return label, TRAIN


def process_file(wav_path: str, window_size: int, seq_length: int, reps: int) -> Iterable[List[float]]:
    rate, data = wav.read(wav_path)

    features: List[List[float]] = []

    index = 0
    while index < len(data):
        # Obtain features using the real component of the fourier transform
        fourier_transform = np.abs(np.fft.fft(data[index:index+window_size]))
        if len(fourier_transform) == window_size:
            normalized_fourier_transform = fourier_transform / (np.max(fourier_transform) + SMALL_NUMBER)
            features.append(normalized_fourier_transform.astype(float).tolist())

        index += window_size

    should_replace = len(features) <= seq_length
    indices = range(len(features))
    for _ in range(reps):
        sampled_indices = np.sort(np.random.choice(indices, size=seq_length, replace=should_replace))
        yield [features[i] for i in sampled_indices]

    return data


def generate_dataset(input_folder: str, output_folder: str, window_size: int, seq_length: int, reps: int, chunk_size: int):
    
    make_dir(output_folder)

    writers = {
        TRAIN: DataWriter(os.path.join(output_folder, TRAIN), file_prefix='data', chunk_size=chunk_size, file_suffix='jsonl.gz'),
        VALID: DataWriter(os.path.join(output_folder, VALID), file_prefix='data', chunk_size=chunk_size, file_suffix='jsonl.gz'),
        TEST: DataWriter(os.path.join(output_folder, TEST), file_prefix='data', chunk_size=chunk_size, file_suffix='jsonl.gz'),
    }

    sample_id = 0
    data_counter: Counter = Counter()

    for data_file in iterate_files(input_folder, pattern=r'.*wav'):
        label, partition = get_label_and_partition(data_file)

        for features in process_file(data_file, window_size=window_size, seq_length=seq_length, reps=reps):
            sample_dict = {
                SAMPLE_ID: sample_id,
                INPUTS: features,
                OUTPUT: label
            }

            writers[partition].add(sample_dict)
            data_counter[partition] += 1

            sample_id += 1

            if sample_id % chunk_size == 0:
                print('Completed {0} samples.'.format(sample_id), end='\r')

    print()

    # Close all data writers
    for writer in writers.values():
        writer.close()

    print('Completed tokenization: {0}'.format(data_counter))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--window-size', type=int, required=True)
    parser.add_argument('--seq-length', type=int, required=True)
    parser.add_argument('--reps', type=int, required=True)
    parser.add_argument('--chunk-size', type=int, default=5000)
    args = parser.parse_args()

    generate_dataset(input_folder=args.input_folder,
                     output_folder=args.output_folder,
                     window_size=args.window_size,
                     seq_length=args.seq_length,
                     reps=args.reps,
                     chunk_size=args.chunk_size)
