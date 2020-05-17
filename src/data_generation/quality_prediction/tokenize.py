import csv
import numpy as np
from datetime import datetime
from argparse import ArgumentParser
from typing import Iterable, Dict, Any

from utils.data_writer import DataWriter
from utils.constants import TIMESTAMP, INPUTS, OUTPUT, SAMPLE_ID


def data_generator(input_path: str, seq_length: int, reps: int) -> Iterable[Dict[str, Any]]:
    with open(input_path, 'r') as input_file:
        reader = csv.reader(input_file, delimiter=',', quotechar='"')
        next(reader)  # Skip the headers

        current_date = None
        output = None
        features: List[List[float]] = []
        sample_id = 0

        for tokens in reader:
            if current_date is None or current_date != tokens[0]:
                if len(features) >= seq_length:

                    indices = list(range(len(features)))
                    seen_permutations = set()
                    for _ in range(reps):
                        sampled_indices = np.sort(np.random.choice(indices, size=seq_length, replace=False))

                        # Prevent duplicates
                        indices_string = ','.join(map(str, sampled_indices))
                        if indices_string in seen_permutations:
                            continue

                        seen_permutations.add(indices_string)

                        sampled_features = [features[i] for i in sampled_indices]
                        date = datetime.strptime(current_date, '%Y-%m-%d %X')

                        sample = {
                            INPUTS: sampled_features,
                            OUTPUT: output,
                            TIMESTAMP: int(date.timestamp()),
                            SAMPLE_ID: sample_id
                        }
                        yield sample

                        sample_id += 1

                current_date = tokens[0]
                features = []

            sample_features = [float(t.replace(',', '.')) for t in tokens[1:-1]]
            output = float(tokens[-1].replace(',', '.'))
            features.append(sample_features)


def tokenize(input_path: str, output_folder: str, seq_length: int, reps: int, chunk_size: int):
    with DataWriter(output_folder, file_prefix='data', chunk_size=chunk_size, file_suffix='jsonl.gz') as writer:

        sample_counter = 0
        for sample in data_generator(input_path, seq_length=seq_length, reps=reps):
            writer.add(sample)
            sample_counter += 1

            if sample_counter % chunk_size == 0:
                print('Completed {0} samples'.format(sample_counter), end='\r')
        print()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--seq-length', type=int, required=True)
    parser.add_argument('--reps', type=int, required=True)
    parser.add_argument('--chunk-size', type=int, default=10000)
    args = parser.parse_args()

    tokenize(args.input_file, args.output_folder, args.seq_length, args.reps, args.chunk_size)
