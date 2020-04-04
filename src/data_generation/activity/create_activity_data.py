import os.path
from argparse import ArgumentParser
from collections import Counter
from typing import Iterable, Dict, Any, Optional

from utils.file_utils import iterate_files, read_by_file_suffix
from utils.data_writer import DataWriter


MONITORS = ['hand', 'chest', 'ankle']
MONITOR_FIELDS = ['temp', 'accel_16_x', 'accel_16_y', 'accel_16_z', 'accel_6_x', 'accel_6_y', 'accel_6_z', 'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z']


def get_label_map() -> Dict[int, int]:
    label_list = read_by_file_suffix('label_map.json')

    label_map: Dict[int, int] = dict()
    for entry in label_list:
        label_map[entry['original_label']] = entry['data_label']

    return label_map


def float_or_none(val: str) -> Optional[float]:
    try:
        return float(val)
    except ValueError:
        return None


def int_or_none(val: str) -> Optional[int]:
    try:
        return int(val)
    except ValueError:
        return None


def tokenize(line: str, label_map: Dict[int, int]) -> Dict[str, Any]:
    tokens = line.split(' ')

    sample: Dict[str, Any] = {
        'timestamp': float_or_none(tokens[0]),
        'label': label_map.get(int_or_none(tokens[1]), 0)
    }

    hand_data = tokens[3:16]
    chest_data = tokens[20:33]
    ankle_data = tokens[37:50]
    data = [hand_data, chest_data, ankle_data]
    for monitor_name, measurements in zip(MONITORS, data):
        for field, val in zip(MONITOR_FIELDS, measurements):
            sample[f'{monitor_name}_{field}'] = float_or_none(val)

    return sample


def read_file(file_path: str, label_map: Dict[int, int]) -> Iterable[Dict[str, Any]]:
    with open(file_path, 'r') as f:
        for line in f:
            sample = tokenize(line, label_map)
            
            should_use = not any((val is None for val in sample.values()))
            if should_use:
                yield sample


def tokenize_dataset(input_folder: str, output_folder: str, file_prefix: str, chunk_size: int, max_num_samples: Optional[int]):

    label_map = get_label_map()
    label_counter: Counter = Counter()

    with DataWriter(output_folder, file_prefix=file_prefix, chunk_size=chunk_size, file_suffix='jsonl.gz') as writer:
        
        data_files = iterate_files(input_folder, pattern=r'.*dat')
        index = 0

        for data_file in data_files:
            for sample in read_file(data_file, label_map):
                writer.add(sample)

                label_counter[sample['label']] += 1
                index += 1

                if index % chunk_size == 0:
                    print(f'Completed {index} samples.', end='\r')

                if max_num_samples is not None and index >= max_num_samples:
                    break

            if max_num_samples is not None and index >= max_num_samples:
                break

        print()
        print(label_counter)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--file-prefix', type=str, default='data')
    parser.add_argument('--chunk-size', type=int, default=5000)
    parser.add_argument('--max-num-samples', type=int)
    args = parser.parse_args()

    assert os.path.exists(args.input_folder), f'The folder {args.input_folder} does not exist!'

    tokenize_dataset(input_folder=args.input_folder,
                     output_folder=args.output_folder,
                     file_prefix=args.file_prefix,
                     chunk_size=args.chunk_size,
                     max_num_samples=args.max_num_samples)
