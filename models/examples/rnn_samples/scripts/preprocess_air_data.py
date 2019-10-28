import csv
from argparse import ArgumentParser
from dpu_utils.utils import ChunkWriter, RichPath
from typing import List, Dict, Any


def try_convert_to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except ValueError:
        return default

def convert_row(record: List[str]):
    return dict(year=record[1],
                month=record[2],
                day=record[3],
                hour=record[4],
                pm=try_convert_to_float(record[5], 0.0),
                dew_point=float(record[6]),
                temp=float(record[7]),
                pressure=float(record[8]),
                wind_direction=record[9],
                cum_wind_speed=float(record[10]),
                cum_snow=float(record[11]),
                cum_rain=float(record[12]))


def process(input_file: str, output_folder: RichPath, chunk_size: int = 10000):
    output_folder.make_as_dir()

    total = 0
    with ChunkWriter(output_folder, file_prefix='data', max_chunk_size=chunk_size, file_suffix='.jsonl.gz', parallel_writers=0) as writer:

        with open(input_file, 'r') as data_file:
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
    parser.add_argument('--input-file', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--chunk-size', type=int, default=10000)
    args = parser.parse_args()

    process(args.input_file, RichPath.create(args.output_folder), args.chunk_size)
