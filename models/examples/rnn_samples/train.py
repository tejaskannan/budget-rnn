from argparse import ArgumentParser
from os.path import join, exists
from utils.hyperparameters import HyperParameters
from rnn_sample_model import RNNSampleModel
from rnn_sample_dataset import RNNSampleDataset
from typing import Optional


def train(data_folder: str, save_folder: str, params_file: str, max_epochs: Optional[int] = None):
    hypers = HyperParameters(params_file)
    model = RNNSampleModel(hyper_parameters=hypers, save_folder=save_folder)

    # Create dataset
    train_folder = join(data_folder, 'train')
    valid_folder = join(data_folder, 'valid')
    test_folder = join(data_folder, 'test')
    dataset = RNNSampleDataset(train_folder, valid_folder, test_folder)

    if max_epochs is not None:
        hypers.epochs = max_epochs

    # Train the model
    model.train(dataset=dataset)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-folders', type=str, nargs='+')
    parser.add_argument('--save-folder', type=str, required=True)
    parser.add_argument('--params-files', type=str, nargs='+')
    parser.add_argument('--testrun', action='store_true')
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

    for data_folder in args.data_folders:
        print(f'Started {data_folder}')
        print('====================')
        for i, params_file in enumerate(args.params_files):
            print(f'Started training model {i+1}/{len(args.params_files)}')
            train(data_folder=data_folder,
                  save_folder=args.save_folder,
                  params_file=params_file,
                  max_epochs=max_epochs)
            print('====================')