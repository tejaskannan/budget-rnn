from argparse import ArgumentParser
from os.path import join, exists
from utils.hyperparameters import HyperParameters
from rnn_sample_model import RNNSampleModel
from rnn_sample_dataset import RNNSampleDataset


def train(data_folder: str, save_folder: str, params_file: str):
    # Validate parameters file
    if not exists(params_file):
        print('The file {args.params_file} does not exist!')
        return

    if not params_file.endswith('.json'):
        print(f'The parameters file must be a JSON.')
        return

    hypers = HyperParameters(params_file)
    model = RNNSampleModel(hyper_parameters=hypers, save_folder=save_folder)

    # Validate data folders
    if not exists(data_folder):
        print(f'The folder {args.data_folder} does not exist!')
        return

    train_folder = join(data_folder, 'train')
    if not exists(train_folder):
        print(f'The folder {train_folder} does not exist!')
        return

    valid_folder = join(data_folder, 'valid')
    if not exists(valid_folder):
        print(f'The folder {valid_folder} does not exist!')
        return

    test_folder = join(data_folder, 'test')
    if not exists(test_folder):
        print(f'The folder {test_folder} does not exist!')
        return

    # Create dataset
    dataset = RNNSampleDataset(train_folder, valid_folder, test_folder)

    # Train the model
    model.train(dataset=dataset)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--save-folder', type=str, required=True)
    parser.add_argument('--params-files', nargs='+', required=True)
    args = parser.parse_args()

    assert args.params_files is not None and len(args.params_files) > 0, f'Must prove at least one set of parameters'

    for i, params_file in enumerate(args.params_files):
        print(f'Started training model {i+1}/{len(args.params_files)}')
        train(data_folder=args.data_folder,
              save_folder=args.save_folder,
              params_file=params_file)
        print('====================')

#    # Validate parameters file
#    assert exists(args.params_file), f'The file {args.params_file} does not exist!'
#    assert args.params_file.endswith('.json'), f'The parameters file must be a JSON.'
#
#    hypers = HyperParameters(args.params_file)
#    model = RNNSampleModel(hyper_parameters=hypers, save_folder=args.save_folder)
#
#    # Validate data folder
#    assert exists(args.data_folder), f'The folder {args.data_folder} does not exist!'
#
#    train_folder = join(args.data_folder, 'train')
#    valid_folder = join(args.data_folder, 'valid')
#    test_folder = join(args.data_folder, 'test')
#
#    # Validate folder of each fold
#    assert exists(train_folder), f'The folder {train_folder} does not exist!'
#    assert exists(valid_folder), f'The folder {valid_folder} does not exist!'
#    assert exists(test_folder), f'The folder {test_folder} does not exist!'
#
#    # Create dataset
#    dataset = RNNSampleDataset(train_folder, valid_folder, test_folder)
#
#    # Train the model
#    model.train(dataset=dataset)
