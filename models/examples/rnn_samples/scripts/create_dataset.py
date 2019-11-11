from argparse import ArgumentParser
from dpu_utils.utils import RichPath, ChunkWriter
from itertools import chain
from typing import List, Dict, Any
from datetime import datetime, timedelta


CHUNK_SIZE = 10000


def parse_date(date_string: str) -> datetime:
    return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%S')


def extract_fields(record: Dict[str, Any], fields: List[str]) -> List[Any]:
    return [record[key] for key in fields]


def create(input_folder: RichPath, output_folder: RichPath, id_field: str, input_fields: List[str], output_fields: List[str], window_size: int, lookahead: int, stride: int):
    output_folder.make_as_dir()

    input_files = input_folder.iterate_filtered_files_in_dir('*.jsonl.gz')
    input_dataset = chain(*(input_file.read_by_file_suffix() for input_file in input_files))

    # Load dataset and keep selected fields
    dataset: Dict[datetime, Any] = dict()
    for record in input_dataset:
        input_values = extract_fields(record, input_fields)
        output_values = extract_fields(record, output_fields)
        sample_id = record[id_field]

        if any((x is None for x in input_values)):
            continue

        if any((x is None for x in output_values)):
            continue

        dataset[parse_date(sample_id)] = dict(inputs=input_values,
                                              output=output_values,
                                              sample_id=sample_id)

    with ChunkWriter(output_folder, file_prefix='data', max_chunk_size=CHUNK_SIZE, file_suffix='.jsonl.gz', parallel_writers=0) as writer:

        # Set the increment for the time field
        increment = timedelta(hours=1)

        total, seen = 0, 0
        for date in dataset.keys():

            seen += 1

            inputs: List[Any] = []
            for i in range(0, window_size):
                time = date + increment * i
                if time in dataset:
                    inputs.append(dataset[time]['inputs'])

            if len(inputs) < window_size:
                continue

            output_time = date + increment * (window_size + lookahead)
            if output_time not in dataset:
                continue

            output = dataset[output_time]['output']

            sample_id = dataset[date]['sample_id']
            record = dict(inputs=inputs, output=output, sample_id=sample_id)
            writer.add(record)
            total += 1

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
    print(f'Seen {seen}')

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
    args = parser.parse_args()

    stride = args.stride if args.stride is not None else int(args.window_size / 2)

    create(input_folder=RichPath.create(args.input_folder),
           output_folder=RichPath.create(args.output_folder),
           id_field=args.id_field,
           input_fields=args.input_fields,
           output_fields=args.output_fields,
           window_size=args.window_size,
           lookahead=args.lookahead,
           stride=stride)
