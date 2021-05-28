import os.path
import numpy as np
import matplotlib.pyplot as plt
import random
from argparse import ArgumentParser
from collections import Counter
from scipy.signal import spectrogram
from typing import Iterable, Dict, Any, Optional

from utils.data_writer import DataWriter
from utils.file_utils import make_dir
from utils.constants import SAMPLE_ID, INPUTS, OUTPUT, TRAIN, TEST, VALID

CHUNK_SIZE = 1000
FFT_WINDOW = 200
SEQ_LENGTH = 20
TRAIN_FRAC = 0.85
SAMPLE_RATE = 2000

LABEL_MAP = {
    'nowhale': 0,
    'rightwhale': 1
}


def data_generator(input_file: str) -> Iterable[Dict[str, Any]]:
    with open(input_file, 'r') as fin:

        is_header = True
        sample_id = 0

        for line in fin:
            line = line.strip()
            if line.lower() == '@data':
                is_header = False
            elif not is_header:
                # Split the line into the input signal and label
                tokens = line.split(',')
                input_data = np.array(list(map(float, tokens[:-1])))
                label = LABEL_MAP[tokens[-1].lower()]

                # Compute the spectrogram. Data sampled a 2KHz
                f, t, Sxx = spectrogram(input_data, fs=SAMPLE_RATE, nperseg=FFT_WINDOW, noverlap=0)
                log_Sxx = np.log(Sxx)

                plt.pcolormesh(t, f, log_Sxx)
                plt.colorbar()
                plt.show()

                input_features = log_Sxx.T.astype(float).tolist()

                if len(input_features) >= SEQ_LENGTH:
                    input_features = input_features[0:SEQ_LENGTH]

                yield {
                    SAMPLE_ID: sample_id,
                    INPUTS: input_features,
                    OUTPUT: label
                }

                sample_id += 1


def write_dataset(input_folder: str, output_folder: str, series: str):
    input_path = os.path.join(input_folder, 'RightWhaleCalls_{0}.arff'.format(series.upper()))

    if series == TRAIN:
        writers = {
            TRAIN: DataWriter(os.path.join(output_folder, TRAIN), file_prefix='data', file_suffix='jsonl.gz', chunk_size=CHUNK_SIZE),
            VALID: DataWriter(os.path.join(output_folder, VALID), file_prefix='data', file_suffix='jsonl.gz', chunk_size=CHUNK_SIZE)
        }

        label_counters = {
            TRAIN: Counter(),
            VALID: Counter()
        }
    else:
        writers = {
            TEST: DataWriter(os.path.join(output_folder, TEST), file_prefix='data', file_suffix='jsonl.gz', chunk_size=CHUNK_SIZE)
        }

        label_counters = {
            TEST: Counter()
        }

    for index, sample in enumerate(data_generator(input_path)):
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
            print('Completed {0} samples'.format(index + 1), end='\r')
    print()

    # Close all writers
    for writer in writers.values():
        writer.close()

    print(label_counters)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    args = parser.parse_args()

    random.seed(42)
    make_dir(args.output_folder)

    print('Starting Training Dataset...')
    write_dataset(args.input_folder, args.output_folder, series=TRAIN)

    print('Starting Test Dataset...')
    write_dataset(args.input_folder, args.output_folder, series=TEST)
