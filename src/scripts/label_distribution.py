from argparse import ArgumentParser
from collections import Counter

from utils.file_utils import read_by_file_suffix, iterate_files
from utils.npz_data_manager import NpzDataManager
from utils.constants import OUTPUT, DATA_FIELDS, SAMPLE_ID


def get_label_distribution(folder: str):
    label_counter: Counter = Counter()

    # Load data and create iterator
    data_manager = NpzDataManager(folder=folder, sample_id_name=SAMPLE_ID, fields=DATA_FIELDS)
    data_manager.load()
    data_iterator = data_manager.iterate(should_shuffle=False, batch_size=100)

    for sample in data_iterator:
        label_counter[float(sample[OUTPUT])] += 1

    print(label_counter)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-folder', type=str, required=True)
    args = parser.parse_args()

    get_label_distribution(args.data_folder)
