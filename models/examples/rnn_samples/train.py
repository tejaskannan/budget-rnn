from argparse import ArgumentParser
from os.path import join, exists
from dpu_utils.utils import RichPath
from datetime import datetime

from utils.hyperparameters import HyperParameters, extract_hyperparameters
from rnn_sample_model import RNNSampleModel
from rnn_sample_dataset import RNNSampleDataset
from typing import Optional, Dict


def train(data_folder: str, save_folder: RichPath, hypers: HyperParameters, max_epochs: Optional[int] = None):
    model = RNNSampleModel(hyper_parameters=hypers, save_folder=save_folder)

    # Create dataset
    train_folder = join(data_folder, 'train')
    valid_folder = join(data_folder, 'valid')
    test_folder = join(data_folder, 'test')
    dataset = RNNSampleDataset(train_folder, valid_folder, test_folder)

    if max_epochs is not None:
        hypers.epochs = max_epochs

    # Train the model
    train_label = model.train(dataset=dataset)

    print('Completed training. Beginning testing...')

    # Test the model
    test_results = model.predict(dataset=dataset,
                                 name=model.name,
                                 test_batch_size=1)

    test_result_file = save_folder.join(f'model-test-log-{train_label}.pkl.gz')
    test_result_file.save_as_compressed_file(test_results)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-folders', type=str, nargs='+')
    parser.add_argument('--save-folder', type=str, required=True)
    parser.add_argument('--params-files', type=str, nargs='+')
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--testrun', action='store_true')
    parser.add_argument('--grid-fields', type=str, nargs='*')
    args = parser.parse_args()

    assert args.params_files is not None and len(args.params_files) > 0, f'Must provide at least one set of parameters'

    max_epochs = 2 if args.testrun else None

    # Validate data folders before training (to fail fast)
    for data_folder in args.data_folders:
        assert exists(data_folder), f'The data folder {data_folder} does not exist!'

        train_folder = join(data_folder, 'train')
        assert exists(train_folder), f'The folder {train_folder} does not exist!'

        valid_folder = join(data_folder, 'valid')
        assert exists(valid_folder), f'The folder {valid_folder} does not exist!'

        test_folder = join(data_folder, 'test')
        assert exists(test_folder), f'The folder {test_folder} does not exist!'

    # Validate params files (to fail fast)
    for params_file in args.params_files:
        assert exists(params_file), f'The file {params_file} does not exist!'
        assert params_file.endswith('.json'), f'The params file must be a JSON.'

    trials = max(args.trials, 1)
    num_models = trials * len(args.params_files)
    grid_fields = args.grid_fields if args.grid_fields is not None and len(args.grid_fields) > 0 else None

    # Create save folder (if necessary)
    base_save_folder = RichPath.create(args.save_folder)
    base_save_folder.make_as_dir()
    current_day = datetime.now().strftime('%d_%m_%Y')
    save_folder = base_save_folder.join(current_day)
    save_folder.make_as_dir()

    for data_folder in args.data_folders:
        print(f'Started {data_folder}')
        print('====================')

        for trial in range(trials):
            for i, params_file in enumerate(args.params_files):
                print(f'Started training model {i+1}/{num_models}')
                hypers = extract_hyperparameters(params_file, search_fields=grid_fields)

                for j, hyperparameters in enumerate(hypers):
                    print(f'Started hyperparameter setting {j+1}/{len(hypers)}')
                    train(data_folder=data_folder,
                          save_folder=save_folder,
                          hypers=hyperparameters,
                          max_epochs=max_epochs)
                    print('==========')
                print('====================')
