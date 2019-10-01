from argparse import ArgumentParser
from os.path import split

from simple_dataset import SimpleDataset
from simple_model import SimpleModel
from utils.hyperparameters import HyperParameters
from utils.file_utils import to_rich_path


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--params-file', type=str, required=True)
    args = parser.parse_args()

    hypers = HyperParameters(args.params_file)

    path_tokens = split(args.model_path)
    model = SimpleModel(hypers, path_tokens[0])
    model_name = path_tokens[1].replace('model-', '')
    model.restore(model_name)

    data_path = to_rich_path(args.data_folder)
    dataset = SimpleDataset(data_path.join('train'), data_path.join('valid'), data_path.join('test'))

    test_sample = {'input': 0.5, 'output': 0.25}
    predictions = model.predict([test_sample], dataset)
    print(predictions)
