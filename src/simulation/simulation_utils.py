import os.path
from typing import Optional, Tuple, Dict, Any

from dataset.dataset import Dataset
from dataset.dataset_factory import get_dataset
from models.model_factory import get_model
from models.base_model import Model
from utils.file_utils import read_by_file_suffix, extract_model_name
from utils.constants import METADATA_PATH, HYPERS_PATH, TRAIN, TEST_LOG_PATH
from utils.hyperparameters import HyperParameters


def make_dataset(model_name: str, save_folder: str, dataset_type: str, dataset_folder: Optional[str]) -> Dataset:
    metadata_file = os.path.join(save_folder, METADATA_PATH.format(model_name))
    metadata = read_by_file_suffix(metadata_file)

    # Infer the dataset
    if dataset_folder is None:
        dataset_folder = os.path.dirname(metadata['data_folders'][TRAIN.upper()])

    # Validate the dataset folder
    assert os.path.exists(dataset_folder), f'The dataset folder {dataset_folder} does not exist!'

    return get_dataset(dataset_type=dataset_type, data_folder=dataset_folder)


def make_model(model_name: str, hypers: HyperParameters, save_folder: str) -> Model:
    model = get_model(hypers, save_folder, is_train=False)
    model.restore(name=model_name, is_train=False, is_frozen=False)
    return model


def get_serialized_info(model_path: str, dataset_folder: Optional[str]) -> Tuple[Model, Dataset, Dict[str, Any]]:
    save_folder, model_file = os.path.split(model_path)

    model_name = extract_model_name(model_file)
    assert model_name is not None, f'Could not extract name from file: {model_file}'

    # Extract hyperparameters
    hypers_path = os.path.join(save_folder, HYPERS_PATH.format(model_name))
    hypers = HyperParameters.create_from_file(hypers_path)

    dataset = make_dataset(model_name, save_folder, hypers.dataset_type, dataset_folder)
    model = make_model(model_name, hypers, save_folder)

    # Get test log
    test_log_path = os.path.join(save_folder, TEST_LOG_PATH.format(model_name))
    assert os.path.exists(test_log_path), f'Must perform model testing before post processing'
    test_log = list(read_by_file_suffix(test_log_path))[0]

    return model, dataset, test_log



