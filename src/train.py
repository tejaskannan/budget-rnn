import os.path
from argparse import ArgumentParser
from datetime import datetime

from utils.hyperparameters import HyperParameters
from utils.constants import TRAIN, VALID, TEST
from utils.file_utils import read_by_file_suffix, make_dir, iterate_files
from models.model_factory import get_model
from dataset.dataset_factory import get_dataset
from dataset.dataset import DataSeries
from typing import Optional, Dict, List
from test import test


def train(data_folder: str, save_folder: str, hypers: HyperParameters, should_print: bool, max_epochs: Optional[int] = None) -> str:
    model = get_model(hypers, save_folder=save_folder, is_train=True)

    # Create dataset
    dataset = get_dataset(hypers.dataset_type, data_folder)

    if max_epochs is not None:
        hypers.epochs = max_epochs

    # Train the model
    train_label = model.train(dataset=dataset, should_print=should_print)

    # Close the dataset files
    dataset.close()

    return train_label


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-folders', type=str, nargs='+')
    parser.add_argument('--save-folder', type=str, required=True)
    parser.add_argument('--params-files', type=str, nargs='+')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--should-print', action='store_true')
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

    # Unpack and Validate params files (to fail fast)
    params_files: List[str] = []
    for params_file in args.params_files:
        if os.path.isdir(params_file):
            params_files.extend(iterate_files(params_file, pattern=r'.*json'))
        else:
            params_files.append(params_file)

    for params_file in params_files:
        assert os.path.exists(params_file), f'The file {params_file} does not exist!'
        assert params_file.endswith('.json'), f'The params file must be a JSON.'

    trials = max(args.trials, 1)
    num_models = trials * len(params_files)

    # Create save folder (if necessary)
    base_save_folder = args.save_folder
    make_dir(base_save_folder)

    # Create date-named folder for better organization
    current_day = datetime.now().strftime('%d_%m_%Y')
    save_folder = os.path.join(base_save_folder, current_day)
    make_dir(save_folder)

    for data_folder in args.data_folders:
        print(f'Started {data_folder}')
        print('====================')

        # Use absolute path to avoid issues with relative referencing during later optimization phases
        data_folder = os.path.abspath(data_folder)

        for trial in range(trials):
            print(f'Starting trial {trial+1}/{trials}')

            for i, params_file in enumerate(params_files):
                print(f'Started training model {i+1}/{num_models}')
                hypers = HyperParameters.create_from_file(params_file)

                print(params_file)

                name = train(data_folder=data_folder,
                             save_folder=save_folder,
                             hypers=hypers,
                             should_print=args.should_print,
                             max_epochs=max_epochs)

                print('==========')
                # Collect Results from the validation set
                test(model_name=name,
                     dataset_folder=data_folder,
                     save_folder=save_folder,
                     hypers=hypers,
                     max_num_batches=None,
                     batch_size=None,
                     series=DataSeries.VALID)

                # Collect Results from the testing set
                test(model_name=name,
                     dataset_folder=data_folder,
                     save_folder=save_folder,
                     hypers=hypers,
                     max_num_batches=None,
                     batch_size=None,
                     series=DataSeries.TEST)
                print('====================')
