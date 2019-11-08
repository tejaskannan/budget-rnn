from argparse import ArgumentParser
from dpu_utils.utils import RichPath, ChunkWriter
from itertools import chain
from typing import List, Dict, Any


CHUNK_SIZE = 10000


def extract_fields(record: Dict[str, Any], fields: List[str]) -> List[Any]:
    return [record[key] for key in fields]


def create(input_folder: RichPath, output_folder: RichPath, id_field: str, input_fields: List[str], output_fields: List[str], window_size: int, lookahead: int, stride: int):
    output_folder.make_as_dir()

    input_files = input_folder.iterate_filtered_files_in_dir('*.jsonl.gz')
    input_dataset = chain(*(input_file.read_by_file_suffix() for input_file in input_files))

    with ChunkWriter(output_folder, file_prefix='data', max_chunk_size=CHUNK_SIZE, file_suffix='.jsonl.gz', parallel_writers=0) as writer:
        chunk: List[Dict[str, Any]] = []

        total = 0
        seen = 0
        for record in input_dataset:
            chunk.append(record)
            seen += 1

            if len(chunk) >= CHUNK_SIZE:
                limit = len(chunk) - window_size - lookahead - 1
                for i in range(0, limit, stride):
                    window = [extract_fields(r, input_fields) for r in chunk[i:i+window_size]]
                    result = extract_fields(chunk[i+window_size+lookahead], output_fields)
                    writer.add(dict(inputs=window, output=result, sample_id=chunk[i][id_field]))
                    total += 1
                chunk = chunk[i:]

        if len(chunk) > 0:
            limit = len(chunk) - window_size - lookahead - 1
            for i in range(0, limit, stride):
                window = [extract_fields(r, input_fields) for r in chunk[i:i+window_size]]
                result = extract_fields(chunk[i+window_size+lookahead], output_fields)
                writer.add(dict(inputs=window, output=result, sample_id=chunk[i][id_field]))
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
