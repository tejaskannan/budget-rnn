import os.path
from argparse import ArgumentParser
from dpu_utils.utils import RichPath
from datetime import datetime

from utils.hyperparameters import HyperParameters
from utils.constants import TRAIN, VALID, TEST
from models.rnn_model import RNNModel
from dataset.rnn_sample_dataset import RNNSampleDataset
from typing import Optional, Dict
from test import test


def train(data_folder: str, save_folder: RichPath, hypers: HyperParameters, max_epochs: Optional[int] = None) -> str:
    model = RNNModel(hypers, save_folder=save_folder)

    # Create dataset
    train_folder = os.path.join(data_folder, TRAIN)
    valid_folder = os.path.join(data_folder, VALID)
    test_folder = os.path.join(data_folder, TEST)
    dataset = RNNSampleDataset(train_folder, valid_folder, test_folder)

    if max_epochs is not None:
        hypers.epochs = max_epochs

    # Train the model
    train_label = model.train(dataset=dataset)
    return train_label


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-folders', type=str, nargs='+')
    parser.add_argument('--save-folder', type=str, required=True)
    parser.add_argument('--params-files', type=str, nargs='+')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--testrun', action='store_true')
    args = parser.parse_args()

    assert args.params_files is not None and len(args.params_files) > 0, f'Must provide at least one set of parameters'

    max_epochs = 1 if args.testrun else None

    # Validate data folders before training (to fail fast)
    for data_folder in args.data_folders:
        assert os.path.exists(data_folder), f'The data folder {data_folder} does not exist!'

        train_folder = os.path.join(data_folder, TRAIN)
        assert os.path.exists(train_folder), f'The folder {train_folder} does not exist!'

        valid_folder = os.path.join(data_folder, VALID)
        assert os.path.exists(valid_folder), f'The folder {valid_folder} does not exist!'

        test_folder = os.path.join(data_folder, TEST)
        assert os.path.exists(test_folder), f'The folder {test_folder} does not exist!'

    # Validate params files (to fail fast)
    for params_file in args.params_files:
        assert os.path.exists(params_file), f'The file {params_file} does not exist!'
        assert params_file.endswith('.json'), f'The params file must be a JSON.'

    trials = max(args.trials, 1)
    num_models = trials * len(args.params_files)

    # Create save folder (if necessary)
    base_save_folder = RichPath.create(args.save_folder)
    base_save_folder.make_as_dir()
    current_day = datetime.now().strftime('%d_%m_%Y')
    save_folder = base_save_folder.join(current_day)
    save_folder.make_as_dir()

    for data_folder in args.data_folders:
        print(f'Started {data_folder}')
        print('====================')

        # Use absolute path to avoid issues with relative referencing during later optimization phases
        data_folder = os.path.abspath(data_folder)

        for trial in range(trials):
            print(f'Starting trial {trial+1}/{trials}')

            for i, params_file in enumerate(args.params_files):
                print(f'Started training model {i+1}/{num_models}')
                hypers = HyperParameters.create_from_file(params_file)

                name = train(data_folder=data_folder,
                             save_folder=save_folder,
                             hypers=hypers,
                             max_epochs=max_epochs)

                print('==========')
                test(model_name=name,
                     dataset_folder=data_folder,
                     save_folder=save_folder,
                     hypers=hypers,
                     max_num_batches=None)
                print('====================')
