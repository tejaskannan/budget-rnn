import os
from argparse import ArgumentParser
from random import random

from utils.data_writer import DataWriter, NpzDataWriter
from utils.constants import SAMPLE_ID, DATA_FIELDS, OUTPUT
from dataset.data_manager import get_data_manager


def sample_dataset(input_folder: str, output_folder: str, zero_frac: float, one_frac: float, chunk_size: int, file_prefix: str, is_npz: bool):

    # Load input data
    extension = 'npz' if is_npz else None
    data_manager = get_data_manager(input_folder, SAMPLE_ID, DATA_FIELDS, extension=extension)
    data_manager.load()
    data_iterator = data_manager.iterate(should_shuffle=False, batch_size=100)
    num_samples = data_manager.length

    if is_npz:
        writer = NpzDataWriter(output_folder, file_prefix=file_prefix, file_suffix='npz', chunk_size=chunk_size, sample_id_name=SAMPLE_ID, data_fields=DATA_FIELDS)
    else:
        writer = DataWriter(output_folder, file_prefix=file_prefix, file_suffix='jsonl.gz', chunk_size=chunk_size)

    for index, sample in enumerate(data_iterator):
        # Sample to omit some labels
        r = random()
        if sample[OUTPUT] == 0 and r > zero_frac:
            continue
        elif sample[OUTPUT] == 1 and r > one_frac:
            continue

        writer.add(sample)
        
        if (index + 1) % chunk_size == 0:
            print(f'Completed {index + 1}/{num_samples} samples.', end='\r')
    print()

    # Write any remaining entries
    writer.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--zero-frac', type=float, required=True)
    parser.add_argument('--one-frac', type=float, required=True)
    parser.add_argument('--chunk-size', type=int, default=5000)
    parser.add_argument('--file-prefix', type=str, default='data')
    parser.add_argument('--npz', action='store_true')
    args = parser.parse_args()

    assert os.path.exists(args.input_folder), f'The folder {args.input_folder} does not exist!'
    assert args.zero_frac >= 0.0 and args.zero_frac <= 1.0, f'The zero fraction must be in [0, 1]'
    assert args.one_frac >= 0.0 and args.one_frac <= 1.0, f'The one fraction must be in [0, 1]'

    sample_dataset(input_folder=args.input_folder,
                   output_folder=args.output_folder,
                   zero_frac=args.zero_frac,
                   one_frac=args.one_frac,
                   chunk_size=args.chunk_size,
                   file_prefix=args.file_prefix,
                   is_npz=args.npz)
