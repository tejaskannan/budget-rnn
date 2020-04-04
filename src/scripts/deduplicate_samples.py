import os
import numpy as np
from argparse import ArgumentParser
from annoy import AnnoyIndex
from typing import Iterable, Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler

from utils.constants import INPUTS, SMALL_NUMBER
from utils.file_utils import read_by_file_suffix, iterate_files
from utils.data_writer import DataWriter


def get_data_iterator(folder: str) -> Iterable[Dict[str, Any]]:
    for data_file in iterate_files(folder, pattern=r'.*jsonl\.gz'):
        for sample in read_by_file_suffix(data_file):
            yield sample


def build_data_scaler(dataset: List[Dict[str, Any]]) -> StandardScaler:
    feature_array: List[List[float]] = []
    for sample in dataset:
        for feature_vector in sample[INPUTS]:
            feature_array.append(feature_vector)

    scaler = StandardScaler()
    scaler.fit(feature_array)
    return scaler


def build_feature_index(dataset: List[Dict[str, Any]], scaler: StandardScaler, num_trees: int) -> AnnoyIndex:
    i = 0
    feature_index = None
    feature_array: List[List[float]] = []
    for sample in dataset:
        for feature_vector in sample[INPUTS]:
            
            # Create the feature index. We do this here to infer the number of dimensions.
            if feature_index is None:
                feature_index = AnnoyIndex(len(feature_vector), 'euclidean')

            scaled_features = scaler.transform([feature_vector])[0]
            feature_index.add_item(i, scaled_features)
            i += 1

    # Build the nearest neighbor index
    feature_index.build(num_trees)

    return feature_index


def deduplicate_samples(input_folder: str, output_folder: str, file_prefix: str, chunk_size: int, num_trees: int, threshold: float):

    print('Loading the dataset...')
    data_iterator = get_data_iterator(input_folder)
    dataset = list(data_iterator)
    print('Loaded the dataset. Retrieving metadata.')

    print('Building feature index...')
    scaler = build_data_scaler(dataset)
    feature_index = build_feature_index(dataset, scaler, num_trees)
    print('Built feature index. Beginning deduplication.')

    with DataWriter(output_folder, file_prefix=file_prefix, chunk_size=chunk_size, file_suffix='jsonl.gz') as writer: 
        total = 0
        num_distinct = 0
        feature_distances: List[float] = []

        for i, sample in enumerate(dataset):
            for features in sample[INPUTS]:
                # Get the 2 closest neighbors because this feature vector is in the index. Thus, the closest vector from a distinct
                # sample should be the 2nd closest globally.
                scaled_features = scaler.transform([features])[0]
                _, distances = feature_index.get_nns_by_vector(scaled_features, n=2, include_distances=True)

                dist = distances[1]
                feature_distances.append(dist)

                if dist >= threshold:
                    writer.add(sample)
                    num_distinct += 1
                    break

            total += 1
            if (i+1) % chunk_size == 0:
                print(f'Completed {i+1} samples.', end='\r')

    distinct_frac = float(num_distinct) / total
    print(f'\nTotal of {total} samples.')
    print(f'Number of distinct samples: {num_distinct} ({distinct_frac:.03f})')

    dist_median = np.median(feature_distances)
    print(f'Median distance to nearest feature: {dist_median}')


if __name__ == '__main__':
    parser = ArgumentParser('Script to deduplicate feature-extracted datasets.')
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--file-prefix', type=str, default='data')
    parser.add_argument('--chunk-size', type=int, default=5000)
    parser.add_argument('--num-trees', type=int, default=10)
    args = parser.parse_args()

    assert os.path.exists(args.input_folder), f'The folder {args.input_folder} does not exist!'
    assert args.threshold > 0, 'Must have a positive threshold'

    deduplicate_samples(input_folder=args.input_folder,
                        output_folder=args.output_folder,
                        file_prefix=args.file_prefix,
                        chunk_size=args.chunk_size,
                        num_trees=args.num_trees,
                        threshold=args.threshold)
