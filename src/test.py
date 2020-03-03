import re
import os
from argparse import ArgumentParser
from dpu_utils.utils import RichPath
from typing import Optional

from models.rnn_model import RNNModel
from dataset.rnn_sample_dataset import RNNSampleDataset
from utils.hyperparameters import HyperParameters
from utils.file_utils import extract_model_name
from train import test


def model_test(path: str, dataset_folder: str, max_num_batches: Optional[int]):
    
    save_folder, model_file = os.path.split(path)
    
    model_name = extract_model_name(model_file)
    if model_name.endswith('-loss'):
        model_name = model_name[:-len('-loss')]
    
    hypers_name = 'model-hyper-params-{0}.pkl.gz'.format(model_name)
    hyperparams_file = os.path.join(save_folder, hypers_name)
    hypers = HyperParameters.create_from_file(hyperparams_file)

    train_folder = os.path.join(dataset_folder, 'train')
    valid_folder = os.path.join(dataset_folder, 'valid')
    test_folder = os.path.join(dataset_folder, 'test')
    dataset = RNNSampleDataset(train_folder, valid_folder, test_folder)

    model = RNNModel(hyper_parameters=hypers, save_folder=save_folder)

    # Build model and restore trainable parameters
    model.restore(name=model_name, is_train=False)

    # Test the model
    print('Starting model testing.')
    test_results = model.predict(dataset=dataset,
                                 test_batch_size=hypers.batch_size,
                                 max_num_batches=max_num_batches)

    test_result_file = RichPath.create(save_folder).join(f'model-test-log-{model_name}.jsonl.gz')
    test_result_file.save_as_compressed_file([test_results])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--dataset-folder', type=str, required=True)
    parser.add_argument('--max-num-batches', type=int)
    args = parser.parse_args()

    model_test(args.model_path, args.dataset_folder, args.max_num_batches)
