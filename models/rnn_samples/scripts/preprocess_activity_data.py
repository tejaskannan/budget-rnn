import os
from argparse import ArgumentParser
from dpu_utils.utils import ChunkWriter, RichPath
from collections import Counter
from typing import List, Dict, Any
from datetime import datetime

from processing_utils import try_convert_to_float


def convert_row(record: List[str]):
    return dict(timestamp=try_convert_to_float(record[0], None),
                label=int(record[1]),
                heart_rate=try_convert_to_float(record[2]),
                hand_accel_16_x=try_convert_to_float(record[3]),
                hand_accel_16_y=try_convert_to_float(record[4]),
                hand_accel_16_z=try_convert_to_float(record[5]),
                hand_accel_6_x=try_convert_to_float(record[6]),
                hand_accel_6_y=try_convert_to_float(record[7]),
                hand_accel_6_z=try_convert_to_float(record[8]),
                hand_gyro_x=try_convert_to_float(record[9]),
                hand_gyro_y=try_convert_to_float(record[10]),
                hand_gyro_z=try_convert_to_float(record[11]),
                hand_mag_x=try_convert_to_float(record[12]),
                hand_mag_y=try_convert_to_float(record[13]),
                hand_mag_z=try_convert_to_float(record[14]))

    
def process(input_path: str, label_counter: Counter, writer: ChunkWriter, chunk_size: int):
    total = 0
    with open(input_path, 'r') as data_file:
        data_generator = iter(data_file)

        for record in data_generator:
            record_tokens = record.strip().split()

            try:
                label = int(record_tokens[1])

                writer.add(convert_row(record_tokens))
                total += 1

                label_counter[label] += 1
            except ValueError:
                pass

            if total % chunk_size == 0:
                print(f'Wrote {total} records so far.', end='\r')

    print()
    print(f'Completed file {path}. Wrote {total} records.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--chunk-size', type=int, default=50000)
    args = parser.parse_args()

    label_counter: Counter = Counter()

    with ChunkWriter(args.output_folder, file_prefix='data', max_chunk_size=args.chunk_size, file_suffix='.jsonl.gz', parallel_writers=0) as writer:

        for file_name in os.listdir(args.input_folder):
            path = os.path.join(args.input_folder, file_name)
            process(path, label_counter, writer, args.chunk_size)

    print(f'Label Distribution: {label_counter}')
