import numpy as np
import os.path
from argparse import ArgumentParser
from annoy import AnnoyIndex
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List, Tuple
from itertools import chain

from models.traditional_model import TraditionalModel
from dataset.data_manager import DataManager, InMemoryDataManager
from utils.hyperparameters import HyperParameters
from utils.file_utils import iterate_files, read_by_file_suffix, extract_model_name
from utils.constants import INPUTS, SAMPLE_ID, OUTPUT, HYPERS_PATH, METADATA_PATH


def restore_model(model_path: str) -> TraditionalModel:
    model_folder, model_file_name = os.path.split(model_path)
    model_name = extract_model_name(model_file_name)

    hypers_file_name = HYPERS_PATH.format(model_name)
    hypers = HyperParameters.create_from_file(os.path.join(model_folder, hypers_file_name))

    model = TraditionalModel(hyper_parameters=hypers, save_folder=model_folder, is_train=False)
    model.restore(name=model_name, is_train=False, is_frozen=False)

    return model


def build_index(data_manager: DataManager, num_trees: int) -> Tuple[AnnoyIndex, MinMaxScaler, StandardScaler]:

    data_features: List[List[float]] = []

    for sample in data_manager.iterate(should_shuffle=False, batch_size=100):
        flattened_features = list(chain(*(sample[INPUTS])))
        data_features.append(flattened_features)

    min_max_scaler = MinMaxScaler()
    scaled_features = min_max_scaler.fit_transform(data_features)

    standard_scaler = StandardScaler()
    standard_scaler.fit(data_features)

    dimensions = len(data_features[0])
    data_index = AnnoyIndex(dimensions, 'euclidean')
    for i, features in enumerate(scaled_features):
        data_index.add_item(i, features)

    data_index.build(n_trees=num_trees)
    return data_index, min_max_scaler, standard_scaler


def generate_new_samples(features: List[float], scaler: MinMaxScaler, data_index: AnnoyIndex, multiplier: int, sample_prob: float, std_factor: float) -> List[List[float]]:
    scaled_features = scaler.transform([features])[0]

    nn_id = data_index.get_nns_by_vector(scaled_features, n=2)[1]  # The 0th index will be this set of features
    nn_features = np.array(data_index.get_item_vector(nn_id))

    std = np.abs(scaled_features - nn_features) / std_factor

    new_samples: List[List[float]] = []
    for _ in range(multiplier - 1):
        sampled_features = np.random.normal(loc=nn_features, scale=std, size=scaled_features.shape)
        random_samples = (np.random.uniform(low=0.0, high=1.0, size=scaled_features.shape) < sample_prob).astype(float)

        new_scaled_features = np.where(random_samples, sampled_features, scaled_features)
        new_features = scaler.inverse_transform([new_scaled_features])[0]

        new_samples.append(new_features)

    return new_samples


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--multiplier', type=int, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--sample-prob', type=float, required=True)
    parser.add_argument('--std-factor', type=float, default=1.0)
    parser.add_argument('--num-trees', type=int, default=16)
    parser.add_argument('--chunk-size', type=int, default=5000)
    args = parser.parse_args()

    assert args.multiplier > 1, 'Must have a multiplier greater than one'
    assert args.sample_prob >= 0.0 and args.sample_prob <= 1.0, 'The sample probability must be in [0, 1]'
    assert args.std_factor > 0, 'The std factor must be positive'

    model = restore_model(args.model_path)

    data_manager = InMemoryDataManager(args.input_folder, sample_id_name=SAMPLE_ID, fields=[INPUTS, OUTPUT], extension='jsonl.gz')
    data_manager.load()

    data_index, min_max_scaler, standard_scaler = build_index(data_manager, num_trees=args.num_trees)

    data_iterator = data_manager.iterate(should_shuffle=False, batch_size=100)
    for _ in range(1):
        sample = next(data_iterator)
        features = list(chain(*(sample[INPUTS])))
        new_samples = generate_new_samples(features, min_max_scaler, data_index, args.multiplier, args.sample_prob, args.std_factor)

        print(new_samples)
