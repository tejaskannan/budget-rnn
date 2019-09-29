import numpy as np
from argparse import ArgumentParser
from dpu_utils.utils import ChunkWriter
from typing import Dict, Callable, List

from utils.file_utils import to_rich_path


def make_sample(input_value: int, fn: Callable[[int], int]) -> Dict[str, int]:
    return {
        'input': int(input_value),
        'output': int(fn(input_value))
    }


def create_dataset(output_folder: str, num_samples: int, train_frac: float, valid_frac: float):
    # Create output directories
    output_path = to_rich_path(output_folder)
    output_path.make_as_dir()

    train_folder = output_path.join('train')
    train_folder.make_as_dir()

    valid_folder = output_path.join('valid')
    valid_folder.make_as_dir()

    test_folder = output_path.join('test')
    test_folder.make_as_dir()

    # Create dataset samples
    dataset: List[Dict[str, int]] = []
    inputs = np.arange(start=int(-num_samples / 2), stop=int(num_samples / 2) + 1, dtype=int)
    for input_value in inputs:
        dataset.append(make_sample(input_value, lambda x: x * x))

    # Shuffle Samples
    np.random.shuffle(dataset)

    train_index = int(train_frac * len(dataset))
    valid_index = int(valid_frac * len(dataset)) + train_index

    train_dataset = dataset[:train_index]
    valid_dataset = dataset[train_index:valid_index]
    test_dataset = dataset[valid_index:]

    train_folder.join('data.jsonl.gz').save_as_compressed_file(train_dataset)
    valid_folder.join('data.jsonl.gz').save_as_compressed_file(valid_dataset)
    test_folder.join('data.jsonl.gz').save_as_compressed_file(test_dataset)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--train-frac', type=float, required=True)
    parser.add_argument('--valid-frac', type=float, required=True)
    parser.add_argument('--num-samples', type=int, required=True)
    args = parser.parse_args()

    create_dataset(args.output_folder, args.num_samples, args.train_frac, args.valid_frac)
