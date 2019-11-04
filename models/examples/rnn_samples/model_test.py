import re
import json
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
from dpu_utils.utils import RichPath
from os.path import split, join, exists
from os import mkdir
from typing import Dict, Tuple, List, Optional, DefaultDict, Any

from rnn_sample_model import RNNSampleModel, TestMetrics
from rnn_sample_dataset import RNNSampleDataset
from utils.hyperparameters import HyperParameters


def extract_model_name(model_file: str) -> str:
    match = re.match(r'^model-([^\.]+)\.ckpt.*$', model_file)
    if not match:
        if model_file.startswith('model-'):
            return model_file[len('model-'):]
        return model_file
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


def plot_results(error_metrics: DefaultDict[str, TestMetrics],
                 latency_metrics: DefaultDict[str, TestMetrics],
                 labels: List[str],
                 prediction_ops: List[str],
                 sample_frac: float,
                 output_folder: Optional[str],
                 test_params: Dict[str, Any]):
    plt.style.use('ggplot')

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))

    # Sample Fraction vs Error
    x = [(i+1) * sample_frac for i in range(len(error_metrics))]
    for i, label in enumerate(labels):
        y = [error_metrics[op][i].mean for op in prediction_ops]
        ax1.errorbar(x=x, y=y, fmt='-o', label=label)

    ax1.legend()
    ax1.set_xlabel('Sample Fraction')
    ax1.set_ylabel('MSE')
    ax1.set_title('Mean Squared Error for Each Sample Fraction')
    ax1.set_xticks(ticks=x)

    # Layer Number vs Latency
    x = [(i+1) * sample_frac for i in range(len(latency_metrics))]
    for i, label in enumerate(labels):
        y = [latency_metrics[op][i].mean for op in prediction_ops]
        yerr = [latency_metrics[op][i].std for op in prediction_ops]
        ax2.errorbar(x, y, fmt='-o', capsize=3.0, label=label)

    ax2.legend()
    ax2.set_xlabel('Sample Fraction')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency for Each Sample Fraction')
    ax2.set_xticks(ticks=x)

    # Error vs Latency
    min_latency, max_latency = 1e10, 0.0
    for i, label in enumerate(labels):
        x = [latency_metrics[op][i].mean for op in prediction_ops]
        y = [error_metrics[op][i].mean for op in prediction_ops]

        ax3.plot(x, y, marker='o', label=label)

        # Update min and max for formating reasons
        max_latency = max(max_latency, np.max(x))
        min_latency = min(min_latency, np.min(x))

    ax3.legend()
    ax3.set_xlabel('Latency (ms)', fontsize=10)
    ax3.set_ylabel('MSE', fontsize=10)
    ax3.set_title('Inference Latency vs Mean Squared Error')
    ax3.set_xlim(left=min_latency - 0.01, right=max_latency + 0.01)

    plt.tight_layout()

    if output_folder is None:
        plt.show()
        return

    output_folder_path = RichPath.create(output_folder)
    output_folder_path.make_as_dir()

    output_folder_name = split(output_folder)[1]
    plot_file = output_folder_path.join(output_folder_name + '.pdf')
    params_file = output_folder_path.join(output_folder_name  + '_params.jsonl.gz')
 
    plt.savefig(plot_file.path)
    params_file.save_as_compressed_file([test_params])

    # For now, we save the metrics as a pickle file because Numpy Arrays
    # are not JSON serializable. This should be changed to compressed
    # JSONL files for readability.
    metrics = dict()
    for i, label in enumerate(labels):
        metrics[label] = {
            'error': [error_metrics[op][i]._asdict() for op in prediction_ops],
            'latency': [latency_metrics[op][i]._asdict() for op in prediction_ops]
        }
    metrics_file = RichPath.create(output_folder).join('metrics.pkl.gz')
    metrics_file.save_as_compressed_file([metrics])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test-params-file', type=str, required=True)
    args = parser.parse_args()

    assert exists(args.test_params_file), f'The file {args.test_params_file} does not exist!'
    with open(args.test_params_file, 'r') as test_params_file:
        test_params = json.load(test_params_file)

    data_folder = test_params.get('data_folder')
    assert data_folder is not None, 'Must prove a data folder.'
    assert exists(data_folder), f'The folder {data_folder} does not exist!'

    # Get and validate data folders
    train_folder = join(data_folder, 'train')
    valid_folder = join(data_folder, 'valid')
    test_folder = join(data_folder, 'test')
    assert exists(train_folder), f'The folder {train_folder} does not exist!'
    assert exists(valid_folder), f'The folder {valid_folder} does not exist!'
    assert exists(test_folder), f'The folder {test_folder} does not exist!'

    dataset = RNNSampleDataset(train_folder, valid_folder, test_folder)

    # Validate the hyperparameters to ensure like-for-like comparisons
    sample_frac: float = 0.0
    for i, model_config in enumerate(test_params['models']):
        # Fetching the hyperparameters checks the existence of the params file
        hypers = HyperParameters(model_config['params_file'])
        if i > 0:
            assert sample_frac == hypers.model_params['sample_frac'], f'Sample Fractions are not equal!'
        else:
            sample_frac = hypers.model_params['sample_frac']

    # Number of model outputs and prediction operations
    num_outputs = int(1.0 / sample_frac)
    prediction_ops = [f'prediction_{i}' for i in range(num_outputs)]

    # Inference parameters
    batch_size, num_batches = test_params.get('batch_size'), test_params.get('num_batches')

    error_metrics: DefaultDict[str, TestMetrics] = defaultdict(list)
    latency_metrics: DefaultDict[str, TestMetrics] = defaultdict(list)
    labels: List[str] = []
    num_models = len(test_params['models'])
    for i, model_params in enumerate(test_params['models']):
        print(f'Starting Model {i+1}/{num_models}.')

        error, latency = evaluate_model(model_params, dataset, batch_size, num_batches)

        labels.append(model_params['model_name'])
        for prediction_op in prediction_ops:
            error_metrics[prediction_op].append(error[prediction_op])
            latency_metrics[prediction_op].append(latency[prediction_op])

        print('============')

    output_folder = test_params.get('output_folder')
    plot_results(error_metrics, latency_metrics,
                 labels=labels, sample_frac=sample_frac,
                 prediction_ops=prediction_ops,
                 output_folder=output_folder,
                 test_params=test_params)
