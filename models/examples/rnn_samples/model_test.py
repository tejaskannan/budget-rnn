import re
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from argparse import ArgumentParser
from collections import defaultdict, OrderedDict
from dpu_utils.utils import RichPath
from os.path import split, join, exists
from os import mkdir
from typing import Dict, Tuple, List, Optional, DefaultDict, Any

from rnn_sample_model import RNNSampleModel
from rnn_sample_dataset import RNNSampleDataset
from testing_utils import TestMetrics
from utils.hyperparameters import HyperParameters


def extract_model_name(model_file: str) -> str:
    match = re.match(r'^model-([^\.]+)\.ckpt.*$', model_file)
    if not match:
        if model_file.startswith('model-'):
            return model_file[len('model-'):]
        return model_file
    return match.group(1)


def evaluate_model(model_params: Dict[str, str], dataset: RNNSampleDataset,
                   batch_size: Optional[int], num_batches: Optional[int]) -> TestMetrics:
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


def get_stat(metrics: TestMetrics, stat_name: str) -> float:
    if stat_name == 'median':
        return metrics.median
    return metrics.mean


def plot_axis(test_metrics: Dict[str, TestMetrics],
              series: str,
              stat_name: str,
              x_values: List[float],
              title: str,
              xlabel: str,
              ylabel: str,
              ax: Axes):
    
    # Sample Fraction vs Squared Error
    for label, metrics in test_metrics.items():
        y = [get_stat(metrics[series][op], stat_name) for op in prediction_ops]
        ax.errorbar(x=x_values, y=y, fmt='-o', label=label)

    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(ticks=x_values)


def plot_results(test_metrics: Dict[str, TestMetrics],
                 prediction_ops: List[str],
                 sample_frac: float,
                 output_folder: Optional[str],
                 stat_name: str,
                 test_params: Dict[str, Any]):
    if len(test_metrics) <= 0:
        raise ValueError('Must provde some metrics to graph.')

    if stat_name not in ('mean', 'median'):
        raise ValueError(f'Unknown aggregate metric {stat_name}.')

    plt.style.use('ggplot')

    sample_fractions = [(i+1) * sample_frac for i in range(len(prediction_ops))]

    # Create axes
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(16, 9))

    # Plot each metric
    plot_axis(test_metrics=test_metrics,
              series='squared_error',
              stat_name=stat_name,
              x_values=sample_fractions,
              title='Mean Squared Error (MSE) for Each Sample Fraction',
              xlabel='Sample Fraction',
              ylabel='MSE',
              ax=ax1)

    plot_axis(test_metrics=test_metrics,
              series='abs_error',
              stat_name=stat_name,
              x_values=sample_fractions,
              title='Mean Absolute Error (MAE) for Each Sample Fraction',
              xlabel='Sample Fraction',
              ylabel='MAE',
              ax=ax2)

    plot_axis(test_metrics=test_metrics,
              series='abs_percentage_error',
              stat_name=stat_name,
              x_values=sample_fractions,
              title='Symmetric Absolute Percentage Error (SMAPE) for Each Sample Fraction',
              xlabel='Sample Fraction',
              ylabel='SMAPE',
              ax=ax3)
    
    plot_axis(test_metrics=test_metrics,
              series='latency',
              stat_name=stat_name,
              x_values=sample_fractions,
              title='Inference Latency for Each Sample Fraction',
              xlabel='Sample Fraction',
              ylabel='Latency (ms)',
              ax=ax4)

    plt.tight_layout()

    if output_folder is None:
        plt.show()
        return

    output_folder_path = RichPath.create(output_folder)
    output_folder_path.make_as_dir()

    output_folder_name = split(output_folder)[1] + '-' + stat_name
    plot_file = output_folder_path.join(output_folder_name + '.pdf')
    params_file = output_folder_path.join(output_folder_name  + '_params.jsonl.gz')
 
    plt.savefig(plot_file.path)
    params_file.save_as_compressed_file([test_params])

    # For now, we save the metrics as a pickle file because Numpy Arrays
    # are not JSON serializable. This should be changed to compressed
    # JSONL files for readability.
    metrics_file = RichPath.create(output_folder).join('metrics.pkl.gz')
    metrics_file.save_as_compressed_file(test_metrics)


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

    metrics: Dict[str, TestMetrics] = dict()
    num_models = len(test_params['models'])
    for i, model_params in enumerate(test_params['models']):
        print(f'Starting Model {i+1}/{num_models}.')

        test_metrics = evaluate_model(model_params, dataset, batch_size, num_batches)
        metrics[model_params['model_name']] = test_metrics

        #for prediction_op in prediction_ops:
        #    error_metrics[prediction_op].append(error[prediction_op])
        #    latency_metrics[prediction_op].append(latency[prediction_op])

        print('============')

    output_folder = test_params.get('output_folder')
    for stat_name in ['mean', 'median']:
        plot_results(test_metrics=metrics,
                     sample_frac=sample_frac,
                     prediction_ops=prediction_ops,
                     output_folder=output_folder,
                     stat_name=stat_name,
                     test_params=test_params)
