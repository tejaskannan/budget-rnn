import re
import json
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from collections import OrderedDict
from os.path import split, join, exists
from typing import Dict, Tuple, List, Optional

from rnn_sample_model import RNNSampleModel
from rnn_sample_dataset import RNNSampleDataset
from utils.hyperparameters import HyperParameters


def extract_model_name(model_file: str) -> str:
    match = re.match(r'model-([^\.]+)\.ckpt.*', model_file)
    if not match:
        return model_file.replace('model-', '')
    return match.group(1)


def evaluate_model(model_params: Dict[str, str], dataset: RNNSampleDataset,
                   batch_size: Optional[int], num_batches: Optional[int]) -> Tuple[OrderedDict, OrderedDict]:
    hypers = HyperParameters(model_params['params_file'])

    path_tokens = split(model_params['model_path'])
    folder, file_name = path_tokens[0], path_tokens[1]
    model = RNNSampleModel(hypers, folder)
    model_name = extract_model_name(file_name)

    model.restore_parameters(model_name)
    model.make(is_train=False)
    model.restore_weights(model_name)

    name = join(folder, f'model-{model_name}')
    metrics = model.predict(dataset, name, batch_size, num_batches)
    return metrics


def plot_results(accuracy_metrics: List[OrderedDict], latency_metrics: List[OrderedDict], labels: List[str], sample_frac: float):
    plt.style.use('ggplot')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))
    fig.tight_layout()

    # Layer Number vs Accuracy
    x = [(i+1) * sample_frac for i in range(len(accuracy_metrics[0]))]
    ys = [list(acc.values()) for acc in accuracy_metrics]
    for y, label in zip(ys, labels):
        ax1.plot(x, y, marker='o', label=label)

    ax1.legend()
    ax1.set_xlabel('Sample Fraction')
    ax1.set_ylabel('MSE')
    ax1.set_title('Mean Squared Error for Each Sample Fraction')
    ax1.set_xticks(ticks=x)

    # Layer Number vs Latency
    x = [(i+1) * sample_frac for i in range(len(latency_metrics[0]))]
    ys = [[v * 1000.0 for v in latency.values()] for latency in latency_metrics]
    for y, label in zip(ys, labels):
        ax2.plot(x, y, marker='o', label=label)

    ax2.legend()
    ax2.set_xlabel('Sample Fraction')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency for Each Sample Fraction')
    ax2.set_xticks(ticks=x)

    # Error vs Latency
    min_latency, max_latency = 1e10, 0.0
    for error, latency, label in zip(accuracy_metrics, latency_metrics, labels):
        x = [v * 1000.0 for v in latency.values()]
        y = list(error.values())
        ax3.plot(x, y, marker='o', label=label)
        max_latency = max(max_latency, np.max(x))
        min_latency = min(min_latency, np.min(x))

    ax3.legend()
    ax3.set_xlabel('Latency (ms)')
    ax3.set_ylabel('MSE')
    ax3.set_title('Inference Latency vs Mean Squared Error')
    ax3.set_xlim(left=min_latency - 0.01, right=max_latency + 0.01)

    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-params-file', type=str, required=True)
    parser.add_argument('--data-folder', type=str, required=True)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-batches', type=int)
    args = parser.parse_args()

    assert exists(args.test_params_file), f'The file {args.test_params_file} does not exist!'
    with open(args.test_params_file, 'r') as test_params_file:
        test_params = json.load(test_params_file)

    # Get and validate data folders
    train_folder = join(args.data_folder, 'train')
    valid_folder = join(args.data_folder, 'valid')
    test_folder = join(args.data_folder, 'test')
    assert exists(train_folder), f'The folder {train_folder} does not exist!'
    assert exists(valid_folder), f'The folder {valid_folder} does not exist!'
    assert exists(test_folder), f'The folder {test_folder} does not exist!'

    dataset = RNNSampleDataset(train_folder, valid_folder, test_folder)
    
    accuracy_metrics: List[Dict[str, float]] = []
    latency_metrics: List[Dict[str, float]] = []
    for model_params in test_params['models']:
        accuracy, latency = evaluate_model(model_params, dataset, args.batch_size, args.num_batches)

        accuracy_metrics.append(accuracy)
        latency_metrics.append(latency)

    model0_hypers = HyperParameters(test_params['models'][0]['params_file'])
    sample_frac = model0_hypers.model_params['sample_frac']
    labels = [model['model_name'] for model in test_params['models']]
    plot_results(accuracy_metrics, latency_metrics, labels, sample_frac)
   # name = join(folder, f'model-{model_name}')
   # metrics = model.predict(dataset, name)

   # print(metrics)
