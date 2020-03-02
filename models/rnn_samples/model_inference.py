import re
from argparse import ArgumentParser
from typing import Optional, Dict
from os.path import join, split

from rnn_sample_model import RNNSampleModel
from rnn_sample_dataset import RNNSampleDataset
from utils.hyperparameters import extract_hyperparameters


def extract_model_name(model_file: str) -> str:
    match = re.match(r'^model-([^\.]+)\.ckpt.*$', model_file)
    if not match:
        if model_file.startswith('model-'):
            return model_file[len('model-'):]
        return model_file
    return match.group(1)


def evaluate_model(model_path: str, params_file: str, dataset: RNNSampleDataset,
                   batch_size: Optional[int], num_batches: Optional[int]):
    hypers = extract_hyperparameters(params_file)[0]

    path_tokens = split(model_path)
    folder, file_name = path_tokens[0], path_tokens[1]
    model = RNNSampleModel(hypers, folder)

    model_name = extract_model_name(file_name)

    model.restore_parameters(model_name)
    model.make(is_train=False)
    model.restore_weights(model_name)

    name = join(folder, f'model-{model_name}')
    metrics = model.predict(dataset, name, batch_size, num_batches)
    return metrics


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--model-params', type=str, required=True)
    parser.add_argument('--dataset-folder', type=str, required=True)
    args = parser.parse_args()

    dataset = RNNSampleDataset(train_folder=join(args.dataset_folder, 'train'),
                               valid_folder=join(args.dataset_folder, 'valid'),
                               test_folder=join(args.dataset_folder, 'test'))

    results = evaluate_model(model_path=args.model_path,
                             params_file=args.model_params,
                             dataset=dataset,
                             batch_size=1,
                             num_batches=1)
    print(results)
