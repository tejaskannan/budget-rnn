import os.path
import numpy as np
from argparse import ArgumentParser
from collections import Counter
from itertools import chain
from typing import Dict, Tuple, Iterable, Any

from utils.data_writer import DataWriter
from utils.constants import INPUTS, OUTPUT, SAMPLE_ID, TIMESTAMP


def load_labels(path: str) -> Dict[Tuple[int, int], int]:
    with open(path, 'r') as label_file:
        label_dict = dict()

        for line in label_file:
            tokens = line.split(',')

            start, end = int(tokens[0].strip()), int(tokens[1].strip())
            label = int(tokens[2].strip())
            label_dict[(start, end)] = label

        return label_dict


def get_label(timestamp: int, label_dict: Dict[Tuple[int, int], int]) -> Tuple[Tuple[int, int], int]:
    for time_range, label in label_dict.items():
        if timestamp >= time_range[0] and timestamp <= time_range[1]:
            return time_range, label
    return None, None


def get_data_generator(input_path: str, window: int, reps: int, label_dict: Dict[Tuple[int, int], int]) -> Iterable[Dict[str, Any]]:
    with open(input_path, 'r') as input_file:
        
        data_window: List[List[float]] = []
        
        current_timestamp = None
        label = None

        for i, line in enumerate(input_file):
            if i == 0:
                continue

            tokens = line.split(',')

            timestamp = int(tokens[0].strip())
            features = [float(t.strip()) for t in tokens[1:]]

            if current_timestamp is None or timestamp <= current_timestamp[0] or timestamp >= current_timestamp[1]:
                if len(data_window) >= window:
                    indices = list(range(len(data_window)))
                    seen = set()

                    for _ in range(reps):
                        sampled_indices = np.sort(np.random.choice(indices, size=window, replace=False))

                        # Remove duplicates
                        sampled_indices_str = ','.join(map(str, sampled_indices))
                        if sampled_indices_str in seen:
                            continue

                        seen.add(sampled_indices_str)

                        sampled_features = [data_window[i] for i in sampled_indices]

                        sample_dict = {
                            INPUTS: sampled_features,
                            OUTPUT: label,
                            TIMESTAMP: timestamp
                        }
                        yield sample_dict

                data_window = []
                current_timestamp, label = get_label(timestamp, label_dict)

            data_window.append(features)


def tokenize_dataset(input_folder: str,
                     output_folder: str,
                     window: int,
                     reps: int,
                     chunk_size: int,
                     label_one_dict: Dict[Tuple[int, int], int],
                     label_two_dict: Dict[Tuple[int, int], int]):

    smartwatch_generator_one = get_data_generator(os.path.join(input_folder, 'measure1_smartwatch_sens.csv'), window, reps, label_one_dict)
    smartwatch_generator_two = get_data_generator(os.path.join(input_folder, 'measure2_smartwatch_sens.csv'), window, reps, label_two_dict)

    smartphone_generator_one = get_data_generator(os.path.join(input_folder, 'measure1_smartphone_sens.csv'), window, reps, label_one_dict)
    smartphone_generator_two = get_data_generator(os.path.join(input_folder, 'measure2_smartphone_sens.csv'), window, reps, label_one_dict)

    data_generator = chain(smartwatch_generator_one, smartwatch_generator_two, smartphone_generator_one, smartphone_generator_two)

    label_distribution: Counter = Counter()

    with DataWriter(output_folder, file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size) as writer:
        sample_id = 0
        for sample in data_generator:
            sample[SAMPLE_ID] = sample_id
            label_distribution[sample[OUTPUT]] += 1

            writer.add(sample)

            sample_id += 1

            if sample_id % chunk_size == 0:
                print('Completed {0} samples.'.format(sample_id), end='\r')

    print()
    print('Total: {0}'.format(sample_id))
    print(label_distribution)

#    with DataWriter(output_folder, file_prefix='data', chunk_size=chunk_size, file_suffix='jsonl.gz') as writer:
        



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--reps', type=int, required=True)
    parser.add_argument('--chunk-size', type=int, default=10000)
    args = parser.parse_args()

    assert os.path.exists(args.input_folder), 'The folder {0} does not exist!'.format(args.input_folder)

    label_one_dict = load_labels(os.path.join(args.input_folder, 'measure1_timestamp_id.csv'))
    label_two_dict = load_labels(os.path.join(args.input_folder, 'measure2_timestamp_id.csv'))

    tokenize_dataset(input_folder=args.input_folder,
                     output_folder=args.output_folder,
                     window=args.window,
                     reps=args.reps,
                     chunk_size=args.chunk_size,
                     label_one_dict=label_one_dict,
                     label_two_dict=label_two_dict)

