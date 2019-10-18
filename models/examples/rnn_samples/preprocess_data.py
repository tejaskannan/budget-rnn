import csv
from argparse import ArgumentParser
from dpu_utils.utils import ChunkWriter, RichPath
from typing import List, Dict, Any


def convert_row(record: List[str]):
    return dict(date=record[0],
                time=record[1],
                global_active_power=float(record[2]),
                global_reactive_power=float(record[3]),
                voltage=float(record[4]),
                global_intensity=float(record[5]),
                sub_metering_one=float(record[6]),
                sub_metering_two=float(record[7]),
                sub_metering_three=float(record[8]))


def process(input_file: str, output_folder: RichPath, chunk_size: int = 10000):
    output_folder.make_as_dir()

    total = 0
    with ChunkWriter(output_folder, file_prefix='data', max_chunk_size=chunk_size, file_suffix='.jsonl.gz', parallel_writers=1) as writer:

        with open(input_file, 'r') as data_file:
            data_generator = iter(data_file)
            next(data_generator)  # Skip the header

            for record in data_generator:
                try:
                    record_tokens = record.strip().split(';')
                    writer.add(convert_row(record_tokens))
                    total += 1
                except ValueError:
                    pass

    print(f'Completed job. Wrote {total} records.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--chunk-size', type=int, default=10000)
    args = parser.parse_args()

    process(args.input_file, RichPath.create(args.output_folder), args.chunk_size)
