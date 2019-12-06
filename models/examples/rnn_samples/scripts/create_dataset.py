from argparse import ArgumentParser
from dpu_utils.utils import RichPath, ChunkWriter
from itertools import chain
from typing import List, Dict, Any
from datetime import datetime, timedelta

from utils.constants import DATE_FORMAT
from processing_utils import parse_date, extract_fields


CHUNK_SIZE = 50000


def create(input_folder: RichPath,
           output_folder: RichPath,
           id_field: str,
           input_fields: List[str],
           output_fields: List[str],
           window_size: int,
           lookahead: int,
           stride: int,
           time_units: str):
    output_folder.make_as_dir()

    input_files = input_folder.iterate_filtered_files_in_dir('*.jsonl.gz')
    input_dataset = chain(*(input_file.read_by_file_suffix() for input_file in input_files))

    print('Loading dataset...')

    # Load dataset and keep selected fields
    dataset: Dict[datetime, Any] = dict()
    min_date, max_date = None, None
    for record in input_dataset:
        input_values = extract_fields(record, input_fields)
        output_values = extract_fields(record, output_fields)
        sample_id = record[id_field]

        if any((x is None for x in input_values)):
            continue

        if any((x is None for x in output_values)):
            continue

        date = parse_date(sample_id)
        dataset[date] = dict(inputs=input_values,
                             output=output_values,
                             sample_id=sample_id)

        if min_date is None or date < min_date:
            min_date = date

        if max_date is None or date > max_date:
            max_date = date

    assert min_date is not None and max_date is not None, 'Must provide at least one valid sample.'

    print('Starting to write dataset.')

    with ChunkWriter(output_folder, file_prefix='data', max_chunk_size=CHUNK_SIZE, file_suffix='.jsonl.gz', parallel_writers=0) as writer:

        # Set the increment and stride for the time field
        time_units = time_units.lower()
        if time_units == 'hour':
            increment = timedelta(hours=1)
            time_stride = timedelta(hours=stride)
        elif time_units == 'minute':
            increment = timedelta(minutes=1)
            time_stride = timedelta(minutes=stride)
        elif time_units == 'seconds':
            increment = timedelta(seconds=1)
            time_stride = timedelta(seconds=stride)
        else:
            raise ValueError(f'Unknown time units: {time_units}')

        approx_dataset_size = int(len(dataset) / stride)

        total = 0
        date = min_date
        while date <= max_date:

            inputs: List[Any] = []
            for i in range(0, window_size):
                time = date + increment * i
                if time in dataset:
                    inputs.append(dataset[time]['inputs'])

            output_time = date + increment * (window_size + lookahead)

            if len(inputs) < window_size or output_time not in dataset:
                date += time_stride
                continue

            sample_id = dataset[date]['sample_id']
            output = dataset[output_time]['output']
            record = dict(inputs=inputs, output=output, sample_id=sample_id, output_time=output_time.isoformat())

            writer.add(record)
            total += 1
            date += time_stride

            if (total + 1) % CHUNK_SIZE == 0:
                print(f'{total+1} of ~{approx_dataset_size} records written.', end='\r')

    print()

    metadata_file = output_folder.join('metadata.jsonl.gz')
    metadata = {
            'input_fields': input_fields,
            'output_fields': output_fields,
            'window_size': window_size,
            'lookahead': lookahead,
            'stride': stride
    }
    metadata_file.save_as_compressed_file([metadata])

    print(f'Wrote a total of {total} records.')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--window-size', type=int, required=True)
    parser.add_argument('--lookahead', type=int, default=0)
    parser.add_argument('--stride', type=int)
    parser.add_argument('--input-fields', type=str, nargs='+')
    parser.add_argument('--output-fields', type=str, nargs='+')
    parser.add_argument('--id-field', type=str, required=True)
    parser.add_argument('--time-units', type=str, required=True, choices=['hour', 'minute', 'second'])
    args = parser.parse_args()

    stride = args.stride if args.stride is not None else int(args.window_size / 2)

    create(input_folder=RichPath.create(args.input_folder),
           output_folder=RichPath.create(args.output_folder),
           id_field=args.id_field,
           input_fields=args.input_fields,
           output_fields=args.output_fields,
           window_size=args.window_size,
           lookahead=args.lookahead,
           stride=stride,
           time_units=args.time_units)
