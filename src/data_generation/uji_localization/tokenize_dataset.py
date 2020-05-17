import os
from argparse import ArgumentParser
from typing import Iterable, Dict, Any

from utils.file_utils import iterate_files



def get_data_files(folder: str) -> Iterable[str]:
    files = os.listdir(folder)

    # If there are no text files, then crawl all subfolders
    if not any((name.endswith('.txt') for name in files)):
        folders = [os.path.join(folder, subfolder) for subfolder in files]
    else:
        folders = [folder]

    for data_folder in folders:
        for data_file in iterate_files(data_folder, pattern='.*txt'):
            yield data_file


def tokenize_data_file(data_file_path: str) -> Iterable[Dict[str, Any]]:
    with open(data_file_path, 'r') as data_file:

        features: List[List[float]] = []
        label = None

        for line in data_file:
            tokens = line.split()
            if len(tokens) == 1:
                label = int(tokens[0][1])  # Label in the format <label>
                break
            else:
                sample_features = list(map(float, tokens[1:]))
                features.append(sample_features)

    

        print(len(features))
        print(label)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    args = parser.parse_args()

    data_files = list(get_data_files(args.input_folder))

    tokenize_data_file(data_files[0])


