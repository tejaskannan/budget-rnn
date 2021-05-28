import os
from argparse import ArgumentParser

from utils.file_utils import read_by_file_suffix, iterate_files
from utils.constants import DATA_FIELDS


def count_samples(data_folder: str, file_type: str, num_fields: int):
    """
    Counts the number of samples in the given archive.
    """
    count = 0
    for data_file in iterate_files(data_folder, pattern=f'.*\.{file_type}'):
        data = read_by_file_suffix(data_file)

        if file_type == 'npz':
            count += int(len(data) / num_fields)
        else:
            count += sum((1 for _ in data))

    print(f'Total number of samples: {count}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--file-type', type=str, choices=['npz', 'jsonl.gz', 'pkl.gz'])
    parser.add_argument('--num-fields', type=int)
    args = parser.parse_args()

    assert os.path.exists(args.data_folder), f'The folder {args.data_folder} does not exist!'

    num_fields = args.num_fields if args.num_fields is not None else len(DATA_FIELDS)
    count_samples(args.data_folder, args.file_type, num_fields)
