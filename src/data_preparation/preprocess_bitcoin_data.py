import csv
from argparse import ArgumentParser
from dpu_utils.utils import ChunkWriter, RichPath
from typing import List, Dict, Any
from datetime import datetime

from processing_utils import try_convert_to_float


def convert_row(record: List[str]) -> Dict[str, Any]:
    date = datetime.strptime(record[1], '%Y-%m-%d %H:%M:%S')
    return dict(date=date.isoformat(),
                open_price=try_convert_to_float(record[3], None),
                high_price=try_convert_to_float(record[4], None),
                low_price=try_convert_to_float(record[5], None),
                close_price=try_convert_to_float(record[6], None),
                volume=try_convert_to_float(record[7], None))


def process(input_folder: RichPath, output_folder: RichPath, chunk_size: int):
    output_folder.make_as_dir()

    total = 0
    with ChunkWriter(output_folder, file_prefix='data', max_chunk_size=chunk_size, file_suffix='.jsonl.gz', parallel_writers=0) as writer:

        for input_file in input_folder.iterate_filtered_files_in_dir('*.csv'):
            with open(input_file.path, 'r') as data_file:
                data_reader = csv.reader(data_file, delimiter=',', quotechar='|')

                data_generator = iter(data_reader)
                next(data_generator)  # Skip the headers
                next(data_generator)

                for record in data_generator:
                    writer.add(convert_row(record))
                    total += 1

                    if total % chunk_size == 0:
                        print(f'Wrote {total} records.', end='\r')

    print(f'Completed job. Wrote {total} records.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--chunk-size', type=int, default=50000)
    args = parser.parse_args()

    process(RichPath.create(args.input_folder), RichPath.create(args.output_folder), args.chunk_size)
