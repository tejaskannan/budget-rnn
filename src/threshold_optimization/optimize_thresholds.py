import os
import numpy as np
from argparse import ArgumentParser
from typing import Dict, List, Optional, Tuple, Any

from models.adaptive_model import AdaptiveModel
from dataset.dataset import DataSeries, Dataset
from dataset.dataset_factory import get_dataset
from utils.hyperparameters import HyperParameters
from utils.constants import METADATA_PATH, HYPERS_PATH, TEST_LOG_PATH
from utils.rnn_utils import get_prediction_name
from utils.testing_utils import ClassificationMetric
from utils.file_utils import extract_model_name, read_by_file_suffix, save_by_file_suffix
from genetic_optimizer import GeneticThresholdOptimizer


def make_dataset(model_name: str, save_folder: str, dataset_type: str, dataset_folder: Optional[str]) -> Dataset:
    metadata_file = os.path.join(save_folder, METADATA_PATH.format(model_name))
    metadata = read_by_file_suffix(metadata_file)

    # Infer the dataset
    if dataset_folder is None:
        dataset_folder = os.path.dirname(metadata['data_folders'][TRAIN.upper()])

    # Validate the dataset folder
    assert os.path.exists(dataset_folder), f'The dataset folder {dataset_folder} does not exist!'

    return get_dataset(dataset_type=dataset_type, data_folder=dataset_folder)


def get_model(model_name: str, hypers: HyperParameters, save_folder: str) -> AdaptiveModel:
    model = AdaptiveModel(hypers, save_folder, is_train=False)
    model.restore(name=model_name, is_train=False, is_frozen=False)
    return model


def get_serialized_info(model_path: str, dataset_folder: Optional[str]) -> Tuple[AdaptiveModel, Dataset, Dict[str, Any]]:
    save_folder, model_file = os.path.split(model_path)

    model_name = extract_model_name(model_file)
    assert model_name is not None, f'Could not extract name from file: {model_file}'

    # Extract hyperparameters
    hypers_path = os.path.join(save_folder, HYPERS_PATH.format(model_name))
    hypers = HyperParameters.create_from_file(hypers_path)

    dataset = make_dataset(model_name, save_folder, hypers.dataset_type, dataset_folder)
    model = get_model(model_name, hypers, save_folder)

    # Get test log
    test_log_path = os.path.join(save_folder, TEST_LOG_PATH.format(model_name))
    assert os.path.exists(test_log_path), f'Must perform model testing before post processing'
    test_log = list(read_by_file_suffix(test_log_path))[0]

    return model, dataset, test_log



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--optimizer-params', type=str, nargs='+')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dataset-folder', type=str)
    args = parser.parse_args()

    # Extract all parameters first to ensure they exist
    params: List[Dict[str, Any]] = []
    for optimizer_params_file in args.optimizer_params:
        assert os.path.exists(optimizer_params_file), f'The file {optimizer_params_file} does not exist'

        optimizer_params = read_by_file_suffix(optimizer_params_file)
        params.append(optimizer_params)

    # Retrieved saved information
    model, dataset, test_log = get_serialized_info(args.model_path, args.dataset_folder)

    prediction_names = [get_prediction_name(i) for i in range(model.num_outputs)]
    flops_per_level = [test_log[name][ClassificationMetric.FLOPS.name] for name in prediction_names]

    for opt_params in params:
        threshold_optimizer = GeneticThresholdOptimizer(model=model, params=opt_params)
        threshold_optimizer.fit(dataset, series=DataSeries.VALID)

        valid_results = threshold_optimizer.score(dataset, series=DataSeries.VALID, flops_per_level=flops_per_level)
        print(valid_results)


