from argparse import ArgumentParser
from random import random
from typing import Iterable, Any

from utils.file_utils import read_by_file_suffix, iterate_files
from utils.data_writer import DataWriter


def get_data_iterator(folder: str) -> Iterable[Any]:
    data_files = iterate_files(folder, pattern='.*jsonl\.gz')
    for data_file in data_files:
        for sample in read_by_file_suffix(data_file):
            yield sample


def sample_dataset(input_folder: str, output_folder: str, chunk_size: int, file_prefix: str, frac: float):
    with DataWriter(output_folder, file_prefix=file_prefix, chunk_size=chunk_size, file_suffix='jsonl.gz') as writer:
        data_iterator = get_data_iterator(input_folder)
        for sample in data_iterator:
            r = random()
            if r < frac:
                writer.add(sample)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--chunk-size', type=int, default=5000)
    parser.add_argument('--file-prefix', type=str, default='data')
    parser.add_argument('--frac', type=float, required=True)
    args = parser.parse_args()

    assert args.frac > 0.0 and args.frac <= 1.0, 'The fraction must be in the range (0.0, 1.0]'

    sample_dataset(input_folder=args.input_folder,
                   output_folder=args.output_folder,
                   chunk_size=args.chunk_size,
                   file_prefix=args.file_prefix,
                   frac=args.frac)
