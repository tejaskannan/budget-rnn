from argparse import ArgumentParser

from utils.file_utils import to_rich_path
from utils.hyperparameters import HyperParameters
from simple_dataset import SimpleDataset
from simple_model import SimpleModel


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--params-file', type=str, required=True)
    parser.add_argument('--save-folder', type=str, required=True)
    args = parser.parse_args()

    output_path = to_rich_path(args.data_folder)
    dataset = SimpleDataset(output_path.join('train'), output_path.join('valid'), output_path.join('test'))

    hypers = HyperParameters(args.params_file)
    model = SimpleModel(hyper_parameters=hypers, save_folder=args.save_folder)
    model.train(dataset)
