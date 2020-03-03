from argparse import ArgumentParser
from dpu_utils.utils import RichPath, ChunkWriter
from collections import deque, Counter

from processing_utils import isNoneOrNaN


def process_file(input_file: RichPath, zero_label: int, one_label: int, window: int, stride: int, label_distribution: Counter, writer: ChunkWriter):

    data_window = deque()
    dataset_count = 0
    stride_count = stride

    for sample in input_file.read_by_file_suffix():

        features: List[float] = []
        for field in sorted(sample.keys()):
            if field != 'timestamp' and field != 'label' and field != 'heart_rate':
                features.append(sample[field])

        # Skip samples with nonetype values
        if any(map(isNoneOrNaN, features)):
            continue

        # Add this sample to the queue
        data_window.append(features)

        if len(data_window) < window:
            continue

        if sample['label'] == zero_label or sample['label'] == one_label and stride_count >= stride:
            data_dict = dict(inputs=list(data_window),
                             output=1 if sample['label'] == one_label else 0,
                             timestamp=sample['timestamp'])

            writer.add(data_dict)

            label_distribution[data_dict['output']] += 1

            dataset_count += 1
            stride_count = 0
        else:
            stride_count += 1

        # Remove the left-most element to enact a sliding window
        data_window.popleft()

    return dataset_count


if __name__ == '__main__':
    parser = ArgumentParser('Script to create windowed datasets for activity prediction.')
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--zero-label', type=int, required=True)
    parser.add_argument('--one-label', type=int, required=True)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--stride', type=int, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    args = parser.parse_args()
    
    input_folder = RichPath.create(args.input_folder)
    assert input_folder.exists(), f'The folder {input_folder} does not exist'

    label_distribution: Counter = Counter()

    with ChunkWriter(args.output_folder, file_prefix='activity-data', max_chunk_size=50000, file_suffix='.jsonl.gz', parallel_writers=0) as writer:
        count = 0
        for input_file in input_folder.iterate_filtered_files_in_dir('*.jsonl.gz'):
            count += process_file(input_file=input_file,
                                  zero_label=args.zero_label,
                                  one_label=args.one_label,
                                  window=args.window,
                                  stride=args.stride,
                                  label_distribution=label_distribution,
                                  writer=writer)
            print(f'Added {count} records to the dataset.', end='\r')
        print()

    print(label_distribution)
