import numpy as np
import os

from argparse import ArgumentParser
from typing import Dict, Iterable, Any

from utils.constants import DATA_FIELDS, DATA_FIELD_FORMAT, SAMPLE_ID, INDEX_FILE
from utils.file_utils import read_by_file_suffix, iterate_files, make_dir, save_by_file_suffix


def create_iterator(folder: str) -> Iterable[Dict[str, Any]]:
    for data_file in iterate_files(folder, pattern=r'.*\.jsonl\.gz'):
        for sample in read_by_file_suffix(data_file):
            yield sample


def convert_dataset(input_folder: str, output_folder: str, file_prefix: str, chunk_size: int):
    # Create the output folder
    make_dir(output_folder)

    # Dictionary to track samples
    data_dict: Dict[str, Any] = dict()

    # Index of sample id to data file
    data_index: Dict[int, int] = dict()

    # Create iterator of input data
    data_iterator = create_iterator(input_folder)

    file_index = 0
    index = 0
    for sample in data_iterator:
        sample_id = sample.get(SAMPLE_ID, index)

        # Rename fields with the sample id
        sample_dict: Dict[str, Any] = dict()
        for field in DATA_FIELDS:
            field_name = DATA_FIELD_FORMAT.format(field, sample_id)
            sample_dict[field_name] = sample[field]

        # Add sample to file dictionary and data index
        data_dict.update(**sample_dict)
        data_index[sample_id] = file_index

        # Save the chunk
        if (index + 1) % chunk_size == 0:
            output_file = os.path.join(output_folder, f'{file_prefix}{file_index:03}.npz')
            np.savez_compressed(output_file, **data_dict)
            data_dict = dict()
            file_index += 1
            print(f'Completed {index + 1} samples.', end='\r')

        index += 1
    print()

    # Save index
    index_file = os.path.join(output_folder, INDEX_FILE)
    save_by_file_suffix(data_index, index_file)

    # Save any remaining data
    if len(data_dict) > 0:
        output_file = os.path.join(output_folder, f'{file_prefix}{file_index:03}.npz')
        np.savez_compressed(output_file, **data_dict)


if __name__ == '__main__':
    parser = ArgumentParser('Script to convert a JSONL dataset to NPZ files.')
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--file-prefix', type=str, default='data')
    parser.add_argument('--chunk-size', type=int, default=10000)
    args = parser.parse_args()

    assert os.path.exists(args.input_folder), f'The folder {args.input_folder} does not exist!'

    convert_dataset(input_folder=args.input_folder,
                    output_folder=args.output_folder,
                    file_prefix=args.file_prefix,
                    chunk_size=args.chunk_size)
