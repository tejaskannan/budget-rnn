import re
from argparse import ArgumentParser
from os.path import split, join, exists

from rnn_sample_model import RNNSampleModel
from rnn_sample_dataset import RNNSampleDataset
from utils.hyperparameters import HyperParameters


def extract_model_name(model_file: str) -> str:
    match = re.match(r'model-([^\.]+)\.ckpt.*', model_file)
    if not match:
        return model_file.replace('model-', '')
    return match.group(1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--params-file', type=str, required=True)
    args = parser.parse_args()

    hypers = HyperParameters(args.params_file)

    path_tokens = split(args.model_path)
    folder, file_name = path_tokens[0], path_tokens[1]
    model = RNNSampleModel(hypers, folder)
    model_name = extract_model_name(file_name)

    model.restore_parameters(model_name)
    model.hypers.model_params['sample_frac'] = hypers.model_params['sample_frac']
    
    model.make(is_train=False)
    model.restore_weights(model_name)

    # Get and validate data folders
    train_folder = join(args.data_folder, 'train')
    valid_folder = join(args.data_folder, 'valid')
    test_folder = join(args.data_folder, 'test')
    assert exists(train_folder), f'The folder {train_folder} does not exist!'
    assert exists(valid_folder), f'The folder {valid_folder} does not exist!'
    assert exists(test_folder), f'The folder {test_folder} does not exist!'

    dataset = RNNSampleDataset(train_folder, valid_folder, test_folder)
    
    name = join(folder, f'model-{model_name}')
    metrics = model.predict(dataset, name)

    print(metrics)
