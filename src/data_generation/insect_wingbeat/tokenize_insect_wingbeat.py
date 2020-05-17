import os.path
import numpy as np
from argparse import ArgumentParser
from collections import Counter
from typing import Iterable, Dict, Any, List

from utils.data_writer import DataWriter
from utils.constants import INPUTS, OUTPUT, SAMPLE_ID


LABEL_MAP = {
    'aedes_female': 0,
    'aedes_male': 1,
    'fruit_flies': 2,
    'house_flies': 3,
    'quinx_female': 4,
    'quinx_male': 5,
    'stigma_female': 6,
    'stigma_male': 7,
    'tarsalis_female': 8,
    'tarsalis_male': 9
}


SEQ_LENGTH = 22
NUM_FEATURES = 7


def generate_samples(features: List[List[float]], label_name: str, window: int) -> Iterable[Dict[str, Any]]:
    num_features = len(features[0])
    while len(features) < window:
        features.append([0 for _ in range(num_features)])

    num_sequences = int(len(features) / window)
    for offset in range(num_sequences):

        window_features = []
        for window_index in range(window):
            index = offset + window_index * num_sequences
            window_features.append(features[index])

        yield {
            INPUTS: window_features,
            OUTPUT: LABEL_MAP[label_name]
        }


def data_generator(input_path: str, window: int) -> Iterable[Dict[str, Any]]:
    with open(input_path, 'r') as input_file:

        has_reached_data = False

        for line in input_file:
            # Skip all lines until the data portion
            if not has_reached_data:
                has_reached_data = (line.strip() == '@data')
                continue

            label_name = None

            line_samples = line.split('\\n')
            features: List[List[float]] = []

            for sample in line_samples:
                sample = sample.strip().replace('\'', '')
                sample_tokens = sample.split(',')

                if len(sample_tokens) == (SEQ_LENGTH + 1):
                    label_name = sample_tokens[-1].lower()

                sample_features = [float(elem) if elem != '?' else 0.0 for i, elem in enumerate(sample_tokens) if i < NUM_FEATURES]
                while len(sample_features) < NUM_FEATURES:
                    sample_features.append(0)  # Pad features with zeros

                assert len(sample_features) == NUM_FEATURES, 'Found feature list with {0} elements.'.format(len(sample_features))
                features.append(sample_features)

            if label_name is not None:
                for sample_dict in generate_samples(features, label_name, window):
                    yield sample_dict


def tokenize(input_path: str, output_folder: str, window: int, chunk_size: int):

    with DataWriter(output_folder, file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size) as writer:

        label_counter: Counter = Counter()

        for sample_id, sample in enumerate(data_generator(input_path, window)):
            sample[SAMPLE_ID] = sample_id

            label_counter[sample[OUTPUT]] += 1
            writer.add(sample)

            if (sample_id + 1) % chunk_size == 0:
                print('Completed {0} samples.'.format(sample_id + 1), end='\r')

        print()

    print('Completed. Total of {0} samples'.format(sample_id + 1))
    print('Label Distribution: {0}'.format(label_counter))    


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--chunk-size', type=int, default=10000)
    args = parser.parse_args()

    assert os.path.exists(args.input_file), 'The file {0} does not exist'.format(args.input_file)
    assert args.window > 0, 'Must have a positive window'
    assert args.chunk_size > 0, 'Must have  a positive chunk size'

    tokenize(input_path=args.input_file,
             output_folder=args.output_folder,
             window=args.window,
             chunk_size=args.chunk_size)
