import csv
from argparse import ArgumentParser
from dpu_utils.utils import ChunkWriter, RichPath
from typing import List, Dict, Any
from datetime import datetime


def try_convert_to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except ValueError:
        return default

def convert_row(record: List[str]):
    year = int(record[1])
    month = int(record[2])
    day = int(record[3])
    hour = int(record[4])
    date = datetime(year=year, month=month, day=day, hour=hour)
    return dict(date=date.isoformat(),
                pm25=try_convert_to_float(record[5], 0.0),
                pm10=try_convert_to_float(record[6], 0.0),
                so2=try_convert_to_float(record[7], 0.0),
                no2=try_convert_to_float(record[8], 0.0),
                co=try_convert_to_float(record[9], 0.0),
                o3=try_convert_to_float(record[10], 0.0),
                temp=try_convert_to_float(record[11], 0.0),
                pressure=try_convert_to_float(record[12], 0.0),
                dew_point=try_convert_to_float(record[13], 0.0),
                rain=try_convert_to_float(record[14], 0.0))

def process(input_folder: RichPath, output_folder: RichPath, chunk_size: int = 10000):
    output_folder.make_as_dir()

    total = 0
    with ChunkWriter(output_folder, file_prefix='data', max_chunk_size=chunk_size, file_suffix='.jsonl.gz', parallel_writers=0) as writer:

        for input_file in input_folder.iterate_filtered_files_in_dir('*.csv'):
            with open(input_file.path, 'r') as data_file:
                data_reader = csv.reader(data_file, delimiter=',', quotechar='|')
                data_generator = iter(data_reader)
                next(data_generator)  # Skip the header

                for record in data_generator:
                    try:
                        writer.add(convert_row(record))
                        total += 1
                    except ValueError:
                        raise

    print(f'Completed job. Wrote {total} records.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--chunk-size', type=int, default=10000)
    args = parser.parse_args()

    process(RichPath.create(args.input_folder), RichPath.create(args.output_folder), args.chunk_size)
