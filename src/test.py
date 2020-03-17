import re
import os
from argparse import ArgumentParser
from dpu_utils.utils import RichPath
from typing import Optional

from models.rnn_model import RNNModel
from dataset.rnn_sample_dataset import RNNSampleDataset
from utils.hyperparameters import HyperParameters
from utils.file_utils import extract_model_name
from utils.constants import HYPERS_PATH, TEST_LOG_PATH, TRAIN, VALID, TEST, METADATA_PATH


def model_test(path: str, max_num_batches: Optional[int]):
    save_folder, model_file = os.path.split(path)

    model_name = extract_model_name(model_file)
    assert model_name is not None, f'Could not extract name from file: {model_file}'

    save_folder = RichPath.create(save_folder)

    # Extract hyperparameters
    hypers_name = HYPERS_PATH.format(model_name)
    hypers = HyperParameters.create_from_file(save_folder.join(hypers_name))

    # Extract data folders
    metadata_file = save_folder.join(METADATA_PATH.format(model_name))
    metadata = metadata_file.read_by_file_suffix()
    train_folder = metadata['data_folders'][TRAIN.upper()]
    print(train_folder)
    dataset_folder, _ = os.path.split(train_folder.path)

    test(model_name=model_name,
         dataset_folder=dataset_folder,
         save_folder=save_folder,
         hypers=hypers,
         max_num_batches=max_num_batches)

def test(model_name: str, dataset_folder: str, save_folder: RichPath, hypers: HyperParameters, max_num_batches: Optional[int]):
    # Create the dataset
    train_folder = os.path.join(dataset_folder, TRAIN)
    valid_folder = os.path.join(dataset_folder, VALID)
    test_folder = os.path.join(dataset_folder, TEST)
    dataset = RNNSampleDataset(train_folder, valid_folder, test_folder)

    model = RNNModel(hyper_parameters=hypers, save_folder=save_folder)

    # Build model and restore trainable parameters
    model.restore(name=model_name, is_train=False)

    # Test the model
    print('Starting model testing...')
    test_results = model.predict(dataset=dataset,
                                 test_batch_size=hypers.batch_size,
                                 max_num_batches=max_num_batches)

    test_result_file = save_folder.join(TEST_LOG_PATH.format(model_name))
    test_result_file.save_as_compressed_file([test_results])
    print('Completed model testing.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--max-num-batches', type=int)
    args = parser.parse_args()

    model_test(args.model_path, args.max_num_batches)
