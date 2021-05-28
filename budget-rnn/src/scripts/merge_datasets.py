from argparse import ArgumentParser
from itertools import chain
from typing import List

from utils.file_utils import iterate_files, read_by_file_suffix
from utils.data_writer import DataWriter
from utils.constants import SAMPLE_ID


def merge_datasets(folders: List[str], output_folder: str, file_prefix: str, file_suffix: str, chunk_size: int):
    with DataWriter(output_folder, file_prefix=file_prefix, file_suffix=file_suffix, chunk_size=chunk_size) as writer:

        data_files = chain(*(iterate_files(folder, pattern=f'.*{file_suffix}') for folder in folders))
        
        sample_id = 0
        for data_file in data_files:
            for sample in read_by_file_suffix(data_file):
                sample[SAMPLE_ID] = sample_id
                writer.add(sample)
                sample_id += 1
                
                if (sample_id + 1) % chunk_size == 0:
                    print('Completed {0} samples.'.format(sample_id + 1), end='\r')
        print()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folders', type=str, nargs='+')
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--file-prefix', type=str, default='data')
    parser.add_argument('--file-suffix', type=str, default='jsonl.gz')
    parser.add_argument('--chunk-size', type=int, default=10000)
    args = parser.parse_args()

    merge_datasets(folders=args.input_folders,
                   output_folder=args.output_folder,
                   file_prefix=args.file_prefix,
                   file_suffix=args.file_suffix,
                   chunk_size=args.chunk_size)
