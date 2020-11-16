import re
import os.path
import math
from argparse import ArgumentParser

from utils.file_utils import read_by_file_suffix, iterate_files
from dataset.dataset_factory import get_dataset
from dataset.dataset import DataSeries


MODEL_NAME_REGEX = re.compile('.*model-train-log-([^-]+-[^0-9]+-.+)_model_best.pkl.gz')
CONTROLLER_FACTOR = 10


if __name__ == '__main__':
    parser = ArgumentParser('Script to count the number of training iterations for various model types')
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--dataset-folder', type=str, required=True)
    args = parser.parse_args()

    dataset = get_dataset(dataset_type='standard', data_folder=args.dataset_folder)

    dataset.dataset[DataSeries.TRAIN].load()
    train_size = dataset.dataset[DataSeries.TRAIN].length

    training_iterations = 0
    model_count = 0

    for train_log_path in iterate_files(args.input_folder, pattern=r'model-train-log.*pkl.gz'):
        match = MODEL_NAME_REGEX.match(train_log_path)
        assert match is not None, 'Could not match {0}'.format(train_log_path)

        # Get the batch size from the hyperparameters
        name = match.group(1)

        save_folder, _ = os.path.split(train_log_path)
        hypers_path = os.path.join(save_folder, 'model-hyper-params-{0}_model_best.pkl.gz'.format(name))
        hypers = read_by_file_suffix(hypers_path)

        batch_size = hypers['batch_size']
        batches_per_epoch = int(math.ceil(train_size / batch_size))

        # Use the number of batches to calculate the total number of iterations
        train_log = read_by_file_suffix(train_log_path)
        train_epochs = len(train_log['loss']['train'])

        training_iterations += train_epochs * batches_per_epoch

        # If there is a controller present, then we use add these iterations to the result
        controller_path = os.path.join(save_folder, 'model-controller-temp-{0}_model_best.pkl.gz'.format(name))
        if os.path.exists(controller_path):
            controller = read_by_file_suffix(controller_path)
            training_iterations += CONTROLLER_FACTOR * controller['training_iters']
            print('Found controller.')

        model_count += 1

    print('Training Iterations: {0}'.format(training_iterations))
