import os.path
import numpy as np
from itertools import chain
from argparse import ArgumentParser
from typing import Iterable, Dict, Any

from utils.constants import TEST, INPUTS, OUTPUT, SAMPLE_ID, TIMESTAMP
from utils.file_utils import iterate_files, read_by_file_suffix
from utils.data_writer import DataWriter


def data_generator(data_folder: str, seq_length: int) -> Iterable[Dict[str, Any]]:
    for data_file in iterate_files(data_folder, pattern=r'.*jsonl.gz'):
        for sample in read_by_file_suffix(data_file):
            indices = list(range(len(sample[INPUTS])))
            sampled_indices = np.sort(np.random.choice(indices, size=seq_length, replace=False))

            sample[INPUTS] = [sample[INPUTS][i] for i in sampled_indices]
            yield sample


def create_testing_set(data_folder: str, seq_length: int, chunk_size: int):
    output_folder = os.path.join(data_folder, '{0}_{1}'.format(TEST, seq_length))

    inputs: List[List[float]] = []
    outputs: List[int] = []

    with DataWriter(output_folder, file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size) as writer:

        test_folder = os.path.join(data_folder, TEST)
        for index, sample in enumerate(data_generator(test_folder, seq_length)):
            writer.add(sample)

            inputs.append(sample[INPUTS])
            outputs.append(sample[OUTPUT])

            if (index + 1) % chunk_size == 0:
                print('Completed {0} samples.'.format(index + 1), end='\r')

        print()
    print('Completed. Writing to text files.')

    # Write to a text file to make it easier for the C implementation
    txt_input_file = os.path.join(data_folder, '{0}_{1}_inputs.txt'.format(TEST, seq_length))
    with open(txt_input_file, 'w') as txt_file:
        for seq in inputs:
            flattened = chain(*seq)
            txt_file.write(' '.join(map(str, flattened)))
            txt_file.write('\n')

    txt_output_file = os.path.join(data_folder, '{0}_{1}_outputs.txt'.format(TEST, seq_length))
    with open(txt_output_file, 'w') as txt_file:
        for label in outputs:
            txt_file.write(str(label))
            txt_file.write('\n')
            

if __name__ == '__main__':
    parser = ArgumentParser('Samples sequences from the testing set to create a concrete set of test samples')
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--seq-length', type=int, required=True)
    parser.add_argument('--chunk-size', type=int, default=5000)
    args = parser.parse_args()

    create_testing_set(args.data_folder, args.seq_length, args.chunk_size)

