from argparse import ArgumentParser
from dpu_utils.utils import RichPath
from collections import Counter


def get_label_distribution(folder: RichPath):
    label_counter: Counter = Counter()

    for data_file in folder.iterate_filtered_files_in_dir('*.jsonl.gz'):
        for sample in data_file.read_by_file_suffix():
            label_counter[sample['output']] += 1


    print(label_counter)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-folder', type=str, required=True)
    args = parser.parse_args()

    folder = RichPath.create(args.data_folder)
    get_label_distribution(folder)
