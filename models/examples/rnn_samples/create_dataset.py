from argparse import ArgumentParser
from dpu_utils.utils import RichPath, ChunkWriter
from itertools import chain
from typing import List, Dict, Any


# ADJUST LEARNING RATES TO ENSURE LOWER LAYERS FIT THE HIGHER LAYERS

CHUNK_SIZE = 10000

def create(input_folder: RichPath, output_folder: RichPath, window_size: int):
    output_folder.make_as_dir()

    input_files = input_folder.iterate_filtered_files_in_dir('*.jsonl.gz')
    input_dataset = chain(*(input_file.read_by_file_suffix() for input_file in input_files))

    with ChunkWriter(output_folder, file_prefix='data', max_chunk_size=CHUNK_SIZE, file_suffix='.jsonl.gz', parallel_writers=1) as writer:
        chunk: List[Dict[str, Any]] = []

        for record in input_dataset:
            chunk.append(record)

            if len(chunk) >= CHUNK_SIZE:
                for i in range(0, len(chunk) - window_size - 1, window_size + 1):
                    window = [[r['global_active_power'], r['voltage']] for r in chunk[i:i+window_size]]
                    result = chunk[i+window_size]['global_active_power']
                    writer.add(dict(input_power=window, output_power=result))
                chunk = chunk[i:]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--window-size', type=int, required=True)
    args = parser.parse_args()

    create(input_folder=RichPath.create(args.input_folder),
           output_folder=RichPath.create(args.output_folder),
           window_size=args.window_size)
