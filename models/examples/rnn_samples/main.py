from argparse import ArgumentParser
from os.path import join, exists
from utils.hyperparameters import HyperParameters
from rnn_sample_model import RNNSampleModel
from rnn_sample_dataset import RNNSampleDataset


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--save-folder', type=str, required=True)
    parser.add_argument('--params-file', type=str, required=True)
    args = parser.parse_args()

    # Validate parameters file
    assert exists(args.params_file), f'The file {args.params_file} does not exist!'
    assert args.params_file.endswith('.json'), f'The parameters file must be a JSON.'

    hypers = HyperParameters(args.params_file)
    model = RNNSampleModel(hyper_parameters=hypers, save_folder=args.save_folder)

    # Validate data folder
    assert exists(args.data_folder), f'The folder {args.data_folder} does not exist!'

    train_folder = join(args.data_folder, 'train')
    valid_folder = join(args.data_folder, 'valid')
    test_folder = join(args.data_folder, 'test')

    # Validate folder of each fold
    assert exists(train_folder), f'The folder {train_folder} does not exist!'
    assert exists(valid_folder), f'The folder {valid_folder} does not exist!'
    assert exists(test_folder), f'The folder {test_folder} does not exist!'

    # Create dataset
    dataset = RNNSampleDataset(train_folder, valid_folder, test_folder)

    # Train the model
    model.train(dataset=dataset)
