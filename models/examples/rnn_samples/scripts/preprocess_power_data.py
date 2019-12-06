import csv
from argparse import ArgumentParser
from dpu_utils.utils import ChunkWriter, RichPath
from typing import List, Dict, Any
from datetime import datetime

from processing_utils import try_convert_to_float

def convert_row(record: List[str]):
    date = datetime.strptime(record[0], '%d/%m/%Y')
    time = datetime.strptime(record[1], '%H:%M:%S')
    full_date = datetime(year=date.year, month=date.month, day=date.day,
                         hour=time.hour, minute=time.minute, second=time.second)

    return dict(date=full_date.isoformat(),
                global_active_power=try_convert_to_float(record[2], None),
                global_reactive_power=try_convert_to_float(record[3], None),
                voltage=try_convert_to_float(record[4], None),
                global_intensity=try_convert_to_float(record[5], None),
                sub_metering_one=try_convert_to_float(record[6], None),
                sub_metering_two=try_convert_to_float(record[7], None),
                sub_metering_three=try_convert_to_float(record[8], None))


def process(input_file: str, output_folder: RichPath, chunk_size: int = 10000):
    output_folder.make_as_dir()

    total = 0
    with ChunkWriter(output_folder, file_prefix='data', max_chunk_size=chunk_size, file_suffix='.jsonl.gz', parallel_writers=0) as writer:

        with open(input_file, 'r') as data_file:
            data_generator = iter(data_file)
            next(data_generator)  # Skip the header

            for record in data_generator:
                record_tokens = record.strip().split(';')
                writer.add(convert_row(record_tokens))
                total += 1

                if total % chunk_size == 0:
                    print(f'Wrote {total} records so far.', end='\r')

    print()
    print(f'Completed job. Wrote {total} records.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--chunk-size', type=int, default=10000)
    args = parser.parse_args()

    process(args.input_file, RichPath.create(args.output_folder), args.chunk_size)
