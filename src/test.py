import re
import os
from argparse import ArgumentParser
from dpu_utils.utils import RichPath

from rnn_sample_model import RNNSampleModel
from rnn_sample_dataset import RNNSampleDataset
from utils.hyperparameters import HyperParameters, extract_hyperparameters
from utils.file_utils import extract_model_name
from train import test


def model_test(path: str, dataset_folder: str):
    
    save_folder, model_file = os.path.split(path)
    
    model_name = extract_model_name(model_file)
    if model_name.endswith('-loss'):
        model_name = model_name[:-len('-loss')]
    
    hypers_name = 'model-hyper-params-{0}.pkl.gz'.format(model_name)
    hyperparams_file = os.path.join(save_folder, hypers_name)
    hypers = extract_hyperparameters(hyperparams_file)[0]

    train_folder = os.path.join(dataset_folder, 'train')
    valid_folder = os.path.join(dataset_folder, 'valid')
    test_folder = os.path.join(dataset_folder, 'test')
    dataset = RNNSampleDataset(train_folder, valid_folder, test_folder)

    model = RNNSampleModel(hyper_parameters=hypers, save_folder=save_folder)

    # Build model
    model.restore_parameters(name=model_name)
    model.make(is_train=False)
    model.restore_weights(name=model_name)

    # Test the model
    print('Starting model testing.')
    test_results = model.predict(dataset=dataset,
                                 name=model.name,
                                 test_batch_size=hypers.batch_size)

    test_result_file = RichPath.create(save_folder).join(f'model-test-log-{model_name}.pkl.gz')
    test_result_file.save_as_compressed_file(test_results)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--dataset-folder', type=str, required=True)
    args = parser.parse_args()

    model_test(args.model_path, args.dataset_folder)
