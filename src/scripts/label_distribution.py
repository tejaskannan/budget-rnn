from argparse import ArgumentParser
from collections import Counter

from utils.file_utils import read_by_file_suffix, iterate_files
from utils.constants import OUTPUT, DATA_FIELDS, SAMPLE_ID
from dataset.data_manager import get_data_manager


def get_label_distribution(folder: str, file_type: str):
    label_counter: Counter = Counter()

    # Load data and create iterator
    data_manager = get_data_manager(folder=folder, sample_id_name=SAMPLE_ID, fields=DATA_FIELDS, extension=file_type)
    data_manager.load()
    data_iterator = data_manager.iterate(should_shuffle=False, batch_size=100)

    total = 0
    for sample in data_iterator:
        label_counter[float(sample[OUTPUT])] += 1
        total += 1

    for key, value in sorted(label_counter.items()):
        frac = float(value) / float(total) * 100
        print('{0}: {1} ({2:.02f})'.format(key, value, frac))

    print(f'Total: {total}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--file-type', type=str, choices=['npz', 'jsonl.gz', 'pkl.gz'])
    args = parser.parse_args()

    get_label_distribution(args.data_folder, args.file_type)
