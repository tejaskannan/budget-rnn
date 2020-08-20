import os.path
from typing import Tuple, Optional

from dataset.dataset import Dataset
from dataset.dataset_factory import get_dataset
from models.base_model import Model
from models.model_factory import get_model

from .constants import TRAIN, METADATA_PATH, HYPERS_PATH
from .file_utils import read_by_file_suffix, extract_model_name
from .hyperparameters import HyperParameters


def make_dataset(model_name: str, save_folder: str, dataset_type: str, dataset_folder: Optional[str]) -> Dataset:
    metadata_file = os.path.join(save_folder, METADATA_PATH.format(model_name))
    metadata = read_by_file_suffix(metadata_file)

    # Infer the dataset
    if dataset_folder is None:
        dataset_folder = os.path.dirname(metadata['data_folders'][TRAIN.upper()])

    # Validate the dataset folder
    assert os.path.exists(dataset_folder), 'The dataset folder {0} does not exist!'.format(dataset_folder)

    return get_dataset(dataset_type=dataset_type, data_folder=dataset_folder)


def make_model(model_name: str, hypers: HyperParameters, save_folder: str) -> Model:
    model = get_model(hypers, save_folder, is_train=False)
    model.restore(name=model_name, is_train=False, is_frozen=False)
    return model


def restore_neural_network(model_path: str, dataset_folder: Optional[str]) -> Tuple[Model, Dataset]:
    save_folder, model_file = os.path.split(model_path)

    model_name = extract_model_name(model_file)
    assert model_name is not None, f'Could not extract name from file: {model_file}'

    # Extract hyperparameters
    hypers_path = os.path.join(save_folder, HYPERS_PATH.format(model_name))
    hypers = HyperParameters.create_from_file(hypers_path)

    dataset = make_dataset(model_name, save_folder, hypers.dataset_type, dataset_folder)
    model = make_model(model_name, hypers, save_folder)

    return model, dataset



