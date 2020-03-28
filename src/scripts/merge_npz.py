import numpy as np
import os
from argparse import ArgumentParser
from typing import List

from utils.constants import SAMPLE_ID, DATA_FIELD_FORMAT, INDEX_FILE
from utils.file_utils import save_by_file_suffix, iterate_files


def merge_npz_files(input_folders: List[str], output_folder: str, file_prefix: str, chunk_size: int, start_index: int, start_id: int):
    """
    Merges all arrays stored in NPZ files in the given input folder into fewer NPZ files
    with unique sample ids.
    """
    # Get all npz files
    data_files: List[str] = []
    for input_folder in input_folders:
        data_files.extend(iterate_files(input_folder, pattern=r'.*\.npz'))

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Dictionary to hold elements
    elements: Dict[str, np.ndarray] = dict()
    data_index: Dict[int, int] = dict()

    # Track index of output file
    file_index = start_index

    # Merge arrays in the given files
    for sample_id, data_file in enumerate(data_files):

        # Copy sample into the new dictionary
        sample = np.load(data_file, mmap_mode='r')
        for field in sample.files:
            field_name = DATA_FIELD_FORMAT.format(field, sample_id + start_id)
            elements[field_name] = sample[field]

        # Add sample to lookup index
        data_index[sample_id + start_id] = file_index

        # Save chunk
        if (sample_id + 1) % chunk_size == 0:
            output_file = os.path.join(output_folder, f'{file_prefix}{file_index:03}.npz')
            np.savez_compressed(output_file, **elements)
            elements = dict()
            file_index += 1

    # Save the reverse index
    index_file = os.path.join(output_folder, INDEX_FILE)
    save_by_file_suffix(data_index, index_file)

    # Clean up any remaining elements
    if len(elements) > 0:
        output_file = os.path.join(output_folder, f'{file_prefix}{file_index:03}.npz')
        np.savez_compressed(output_file, **elements)


if __name__ == '__main__':
    parser = ArgumentParser('Utility script to merge NPZ files')
    parser.add_argument('--input-folders', type=str, nargs='+')
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--file-prefix', type=str, default='data')
    parser.add_argument('--chunk-size', type=int, default=1000)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--start-id', type=int, default=0)
    args = parser.parse_args()

    for input_folder in args.input_folders:
        assert input_folder, f'The folder {input_folder} does not exist!'

    merge_npz_files(args.input_folders, args.output_folder, args.file_prefix, args.chunk_size, args.start_index, args.start_id)
