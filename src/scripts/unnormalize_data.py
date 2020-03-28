import numpy as np
import os
from argparse import ArgumentParser
from typing import Dict, Any

from utils.constants import INDEX_FILE, INPUTS
from utils.file_utils import iterate_files, make_dir, read_by_file_suffix, save_by_file_suffix


def unnormalize_data(input_folder: str, output_folder: str):
    
    normalize_array = np.array([[540, 960, 540, 960, 540, 960, 540, 960]])

    for data_file in iterate_files(input_folder, pattern=r'.*\.npz'):
        dataset = np.load(data_file)

        updated_dataset: Dict[str, Any] = dict()
        for key in dataset.keys():
            if key.startswith(INPUTS):
                updated_dataset[key] = dataset[key] * normalize_array
            else:
                updated_dataset[key] = dataset[key]

        _, data_file_name = os.path.split(data_file)
        np.savez_compressed(os.path.join(output_folder, data_file_name), **updated_dataset)

    # Copy the data index
    data_index = read_by_file_suffix(os.path.join(input_folder, INDEX_FILE))
    save_by_file_suffix(data_index, os.path.join(output_folder, INDEX_FILE))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    args = parser.parse_args()

    assert os.path.exists(args.input_folder), f'The folder {args.input_folder} does not exist!'

    make_dir(args.output_folder)  # Create the output folder (if needed)
    unnormalize_data(args.input_folder, args.output_folder)
