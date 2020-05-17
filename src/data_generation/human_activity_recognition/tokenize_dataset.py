import os.path
from argparse import ArgumentParser
from collections import Counter, deque
from typing import Iterable, Dict, Any, List, Tuple
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils.data_writer import DataWriter
from utils.constants import INPUTS, OUTPUT, SAMPLE_ID, TRAIN, TEST
from utils.file_utils import read_by_file_suffix, make_dir


def get_data_iterator(folder: str, series: str) -> Iterable[Dict[str, Any]]:
    with open(os.path.join(folder, 'X_{0}.txt'.format(series)), 'r') as features_file, \
        open(os.path.join(folder, 'y_{0}.txt'.format(series)), 'r') as labels_file:

        for features, label in zip(features_file, labels_file):
            tokens = [t.strip() for t in features.split(' ') if len(t.strip()) > 0]

            try:
                feature_values = [float(val) for val in tokens]
                sample = {
                    INPUTS: feature_values,
                    OUTPUT: int(label)
                }

                yield sample
            except ValueError:
                raise


def majority(elements: List[int]) -> int:
    elem_counter: Counter = Counter()
    for x in elements:
        elem_counter[x] += 1
    return elem_counter.most_common(1)[0][0]


def create_pca(input_folder: str, variance: str) -> Tuple[PCA, StandardScaler]:
    # Read and scale data
    input_features: List[List[float]] = []
    for sample in get_data_iterator(input_folder, TRAIN):
        input_features.append(sample[INPUTS])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(input_features)

    pca = PCA(variance)
    pca.fit(scaled_features)
    return pca, scaler


def tokenize_data(input_folder: str, output_folder: str, window: int, stride: int, chunk_size: int, variance: float):

    pca, scaler = create_pca(input_folder, variance)

    make_dir(output_folder)
    with DataWriter(os.path.join(output_folder, TRAIN), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size) as train_writer, \
        DataWriter(os.path.join(output_folder, TEST), file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size) as test_writer:
        
        label_counter = Counter()
        data_window: deque = deque()

        stride_counter = stride
        sample_id = 0
        for series, writer in zip([TRAIN, TEST], [train_writer, test_writer]):
            for sample in get_data_iterator(input_folder, series):
                data_window.append(sample)
                
                # Remove excess data entries
                while len(data_window) > window:
                    data_window.popleft()
                
                stride_counter += 1

                # Only write when the stride is fully reached
                if stride_counter >= stride and len(data_window) == window: 

                    label = majority([elem[OUTPUT] for elem in data_window])
                    features = [deepcopy(elem[INPUTS]) for elem in data_window]

                    scaled_features = scaler.transform(features)
                    transformed_features = pca.transform(scaled_features)

                    sample_dict = {
                        SAMPLE_ID: sample_id,
                        INPUTS: transformed_features.astype(float).tolist(),
                        OUTPUT: label
                    }

                    writer.add(sample_dict)
                    stride_counter = 0
                    sample_id += 1
                    label_counter[label] += 1

                if (sample_id + 1) % chunk_size == 0:
                    print('Completed {0} samples.'.format(sample_id + 1), end='\r')
     
        print()
        print(label_counter)


if __name__ == '__main__':
    parser = ArgumentParser('Script to tokenize Eye sensor dataset.')
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--window', type=int, required=True)
    parser.add_argument('--stride', type=int, required=True)
    parser.add_argument('--chunk-size', type=int, required=True)
    parser.add_argument('--variance', type=float, default=0.95)
    args = parser.parse_args()

    tokenize_data(input_folder=args.input_folder,
                  output_folder=args.output_folder,
                  window=args.window,
                  stride=args.stride,
                  chunk_size=args.chunk_size,
                  variance=args.variance)
