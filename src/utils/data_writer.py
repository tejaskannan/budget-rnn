import numpy as np
import os
from argparse import ArgumentParser

from .constants import SAMPLE_ID, DATA_FIELD_FORMAT


def merge_npz_files(input_folder: str, output_folder: str, file_prefix: str, chunk_size: int):
    """
    Merges all arrays stored in NPZ files in the given input folder into fewer NPZ files
    with unique sample ids.
    """
    # Get all npz files
    data_files = [os.path.join(input_folder, name) for name in os.listdir(input_folder) if name.endswith('.npz')]

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Dictionary to hold elements
    elements: Dict[str, np.ndarray] = dict()

    # Track index of output file
    file_index = 0

    # Merge arrays in the given files
    for sample_id, data_file in enumerate(data_files):

        # Copy sample into the new dictionary
        sample = np.load(data_file, mmap_mode='r')
        for field in sample.files:
            field_name = DATA_FIELD_FORMAT.format(field, sample_id)
            elements[field_name] = sample[field]

        # Save chunk
        if (sample_id + 1) % chunk_size == 0:
            output_file = os.path.join(output_folder, f'{file_prefix}{file_index:03}.npz')
            np.savez_compressed(output_file, **elements)
            elements = dict()
            file_index += 1

    # Clean up any remaining elements
    if len(elements) > 0:
        output_file = os.path.join(output_folder, f'{file_prefix}{file_index:03}.npz')
        np.savez_compressed(output_file, **elements)


if __name__ == '__main__':
    parser = ArgumentParser('Utility script to merge NPZ files')
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--file-prefix', type=str, default='data')
    parser.add_argument('--chunk-size', type=int, default=1000)
    args = parser.parse_args()

    assert os.path.exists(args.input_folder), f'The folder {args.input_folder} does not exist!'

    merge_npz_files(args.input_folder, args.output_folder, args.file_prefix, args.chunk_size)
