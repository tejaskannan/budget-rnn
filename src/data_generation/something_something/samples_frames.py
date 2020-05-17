import numpy as np
from argparse import ArgumentParser
from typing import Iterable, Dict, Any, List

from utils.constants import INPUTS, OUTPUT, SAMPLE_ID
from utils.file_utils import iterate_files, read_by_file_suffix
from utils.data_writer import DataWriter
from process_videos import ORIGINAL_ID



def get_data_generator(input_folder: str) -> Iterable[Dict[str, Any]]:
    for data_file in iterate_files(input_folder, pattern=r'.*jsonl.gz'):
        for sample in read_by_file_suffix(data_file):
            yield sample


def round_data(frame: List[List[List[float]]], num_decimals: int) -> List[List[List[float]]]:
    return np.round(frame, num_decimals).astype(float).tolist()


def generate_samples(input_folder: str, output_folder: str, seq_length: int, reps: int, num_decimals: int, chunk_size: int):
    data_generator = get_data_generator(input_folder)    

    with DataWriter(output_folder, file_prefix='data', chunk_size=chunk_size, file_suffix='jsonl.gz') as writer:

        sample_id = 0
        for sample_num, sample in enumerate(data_generator):
            input_frames = sample[INPUTS]
            frame_indices = np.arange(start=0, stop=len(input_frames))

            # Skip sequences that are too short
            if len(input_frames) < seq_length:
                continue

            for _ in range(reps):
                chosen_indices = np.sort(np.random.choice(frame_indices, size=seq_length, replace=False))
                frames = [round_data(input_frames[i], num_decimals) for i in chosen_indices]

                new_sample = {
                    INPUTS: frames,
                    OUTPUT: sample[OUTPUT],
                    SAMPLE_ID: sample_id,
                    ORIGINAL_ID: sample[ORIGINAL_ID]
        
                }

                sample_id += 1
                writer.add(new_sample)

            if (sample_num + 1) % chunk_size == 0:
                print('Completed original {0} samples'.format(sample_num), end='\r')

    print()
    print('Completed Processing. Total of {0} samples in the dataset'.format(sample_id))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--seq-length', type=int, required=True)
    parser.add_argument('--reps', type=int, required=True)
    parser.add_argument('--num-decimals', type=int, default=4)
    parser.add_argument('--chunk-size', type=int, default=1000)
    args = parser.parse_args()

    assert args.seq_length > 0, 'Must have a positive sequence length'
    assert args.reps > 0, 'Must have a positive number of repetitions'

    generate_samples(input_folder=args.input_folder,
                     output_folder=args.output_folder,
                     seq_length=args.seq_length,
                     reps=args.reps,
                     num_decimals=args.num_decimals,
                     chunk_size=args.chunk_size)
