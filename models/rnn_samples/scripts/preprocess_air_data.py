import csv
from argparse import ArgumentParser
from dpu_utils.utils import ChunkWriter, RichPath
from typing import List, Dict, Any
from datetime import datetime

from processing_utils import try_convert_to_float

def convert_row(record: List[str]):
    year = int(record[1])
    month = int(record[2])
    day = int(record[3])
    hour = int(record[4])
    date = datetime(year=year, month=month, day=day, hour=hour)
    return dict(date=date.isoformat(),
                pm25=try_convert_to_float(record[5], None),
                pm10=try_convert_to_float(record[6], None),
                so2=try_convert_to_float(record[7], None),
                no2=try_convert_to_float(record[8], None),
                co=try_convert_to_float(record[9], None),
                o3=try_convert_to_float(record[10], None),
                temp=try_convert_to_float(record[11], None),
                pressure=try_convert_to_float(record[12], None),
                dew_point=try_convert_to_float(record[13], None),
                rain=try_convert_to_float(record[14], None),
                wind_speed=try_convert_to_float(record[16], None))

def process(input_folder: RichPath, output_folder: RichPath, chunk_size: int = 50000):
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

                    if total % chunk_size == 0:
                        print(f'Completed {total} records.', end='\r')

    print()
    print(f'Completed job. Wrote {total} records.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--chunk-size', type=int, default=50000)
    args = parser.parse_args()

    process(RichPath.create(args.input_folder), RichPath.create(args.output_folder), args.chunk_size)
