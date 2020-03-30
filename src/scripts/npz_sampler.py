import numpy as np
import os
from argparse import ArgumentParser
from typing import Dict, List, Any

from utils.constants import DATA_FIELD_FORMAT, SAMPLE_ID, OUTPUT, DATA_FIELDS, INDEX_FILE
from utils.file_utils import make_dir, save_by_file_suffix
from dataset.data_manager import NpzDataManager


def sample_dataset(input_folder: str, output_folder: str, zero_frac: float, one_frac: float, chunk_size: int, file_prefix: str):
    # Create the output folder
    make_dir(output_folder)

    # Load input dataset
    data_manager = NpzDataManager(input_folder, SAMPLE_ID, DATA_FIELDS)
    data_manager.load()
    data_iterator = data_manager.iterate(should_shuffle=False, batch_size=100)
    num_samples = data_manager.length

    data_dict: Dict[str, Any] = dict()
    data_index: Dict[int, int] = dict()

    file_index = 0
    count = 0

    for index, sample in enumerate(data_iterator):

        # Sample to omit some labels
        r = np.random.uniform(low=0.0, high=1.0)
        if sample[OUTPUT] == 0 and r > zero_frac:
            continue
        elif sample[OUTPUT] == 1 and r > one_frac:
            continue

        sample_dict: Dict[str, Any] = dict()
        sample_id = sample[SAMPLE_ID]
        for field in DATA_FIELDS:
            field_name = DATA_FIELD_FORMAT.format(field, sample_id)
            sample_dict[field_name] = sample[field]

        data_dict.update(**sample_dict)
        data_index[sample_id] = file_index
        count += 1

        if count % chunk_size == 0:
            output_file = os.path.join(output_folder, f'{file_prefix}{file_index:03}.npz')
            np.savez_compressed(output_file, **data_dict)
            file_index += 1
            data_dict = dict()

            print(f'Completed {index + 1}/{num_samples} samples.', end='\r')

    print()

    # Save the index
    index_file = os.path.join(output_folder, INDEX_FILE)
    save_by_file_suffix(data_index, index_file)

    if len(data_dict) > 0:
        output_file = os.path.join(output_folder, f'{file_prefix}{file_index:03}.npz')
        np.savez_compressed(output_file, **data_dict)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--zero-frac', type=float, required=True)
    parser.add_argument('--one-frac', type=float, required=True)
    parser.add_argument('--chunk-size', type=int, default=5000)
    parser.add_argument('--file-prefix', type=str, default='data')
    args = parser.parse_args()

    assert os.path.exists(args.input_folder), f'The folder {args.input_folder} does not exist!'
    assert args.zero_frac >= 0.0 and args.zero_frac <= 1.0, f'The zero fraction must be in [0, 1]'
    assert args.one_frac >= 0.0 and args.one_frac <= 1.0, f'The one fraction must be in [0, 1]'

    sample_dataset(args.input_folder, args.output_folder, args.zero_frac, args.one_frac, args.chunk_size, args.file_prefix)
