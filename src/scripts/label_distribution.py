from argparse import ArgumentParser
from collections import Counter

from utils.file_utils import read_by_file_suffix, iterate_files
from utils.constants import OUTPUT, DATA_FIELDS, SAMPLE_ID
from dataset.data_manager import get_data_manager


def get_label_distribution(folder: str, is_npz: bool):
    label_counter: Counter = Counter()

    # Load data and create iterator
    extension = 'npz' if is_npz else None
    data_manager = get_data_manager(folder=folder, sample_id_name=SAMPLE_ID, fields=DATA_FIELDS, extension=extension)
    data_manager.load()
    data_iterator = data_manager.iterate(should_shuffle=False, batch_size=100)

    for sample in data_iterator:
        label_counter[float(sample[OUTPUT])] += 1

    print(label_counter)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--npz', action='store_true')
    args = parser.parse_args()

    get_label_distribution(args.data_folder, is_npz=args.npz)
