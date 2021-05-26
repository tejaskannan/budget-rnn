import os.path
import numpy as np
from itertools import chain
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from typing import Iterable, Dict, Any, List

from utils.constants import TEST, INPUTS, OUTPUT, SAMPLE_ID, TIMESTAMP, TRAIN
from utils.file_utils import iterate_files, read_by_file_suffix


def data_generator(data_folder: str) -> Iterable[Dict[str, Any]]:
    """
    Iterates through all samples in the given data folder.
    """
    for data_file in iterate_files(data_folder, pattern=r'.*jsonl.gz'):
        for sample in read_by_file_suffix(data_file):
            yield sample


def fit_input_scaler(data_folder: str) -> StandardScaler:
    """
    Fits the data normalization object.
    """
    inputs = [sample[INPUTS] for sample in data_generator(data_folder)]

    num_features = len(inputs[0][0])
    input_array = np.array(inputs).reshape(-1, num_features)  # Reshape to a 2D office

    scaler = StandardScaler().fit(input_array)
    return scaler


def to_fixed_point(val: float, precision: int) -> int:
    """
    Converts the given value to fixed point representation with the provided precision.
    """
    return int(val * (1 << precision))


def create_testing_set(data_folder: str, precision: int):
    """
    Converts the given dataset to a format easily read by the C implementation.

    Args:
        data_folder: The folder containing the (jsonl.gz) data files.
        precision: The fixed point precision.
    """
    assert precision < 16 and precision > 0, 'Precision must be in the range [1, 15]'

    output_folder = os.path.join(data_folder, '{0}_{1}'.format(TEST, precision))

    input_scaler = fit_input_scaler(os.path.join(data_folder, TRAIN))

    inputs: List[List[List[int]]] = []
    outputs: List[int] = []

    for sample in data_generator(os.path.join(data_folder, TEST)):
        # Scale inputs and then convert to fixed point
        scaled_inputs = input_scaler.transform(sample[INPUTS])
        fixed_point_inputs = [list(map(lambda x: to_fixed_point(x, precision), features)) for features in scaled_inputs]

        inputs.append(fixed_point_inputs)
        outputs.append(sample[OUTPUT])

    # Write to a text file to make it easier for the C implementation
    txt_input_file = os.path.join(data_folder, '{0}_{1}_inputs.txt'.format(TEST, precision))
    with open(txt_input_file, 'w') as txt_file:
        for seq in inputs:
            flattened = chain(*seq)
            txt_file.write(' '.join(map(str, flattened)))
            txt_file.write('\n')

    txt_output_file = os.path.join(data_folder, '{0}_{1}_outputs.txt'.format(TEST, precision))
    with open(txt_output_file, 'w') as txt_file:
        for label in outputs:
            txt_file.write(str(label))
            txt_file.write('\n')


if __name__ == '__main__':
    parser = ArgumentParser('Samples sequences from the testing set to create a concrete set of test samples')
    parser.add_argument('--data-folder', type=str, required=True, help='The folder containing the data files to convert')
    parser.add_argument('--precision', type=int, required=True, help='The fixed point precision of the feature values')
    args = parser.parse_args()

    create_testing_set(args.data_folder, precision=args.precision)
