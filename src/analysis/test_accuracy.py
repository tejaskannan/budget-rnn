import re
import os
import time
import numpy as np
from argparse import ArgumentParser
from dpu_utils.utils import RichPath
from collections import defaultdict
from typing import Optional

from dataset.dataset import DataSeries
from dataset.rnn_sample_dataset import RNNSampleDataset
from models.rnn_model import RNNModel
from utils.hyperparameters import HyperParameters, extract_hyperparameters
from utils.constants import SMALL_NUMBER
from train import test


def extract_model_name(model_file: str) -> str:
    match = re.match(r'^model-([^\.]+)\.ckpt.*$', model_file)
    if not match:
        if model_file.startswith('model-'):
            return model_file[len('model-'):]
        return model_file
    return match.group(1)


def test_accuracy(model: RNNSampleModel, dataset: RNNSampleDataset, batch_size: int, max_num_batches: Optional[int]):
    test_batch_generator = dataset.minibatch_generator(series=DataSeries.TEST,
                                                       batch_size=batch_size,
                                                       metadata=model.metadata,
                                                       should_shuffle=False,
                                                       drop_incomplete_batches=True)
    computed_levels: List[float] = []
    accuracy_dict: Dict[str, List[float]] = defaultdict(list)
    latency_dict: Dict[str, List[float]] = defaultdict(list)
    precision_dict: Dict[str, List[float]] = defaultdict(list)
    recall_dict: Dict[str, List[float]] = defaultdict(list)

    for batch_num, batch in enumerate(test_batch_generator):
        feed_dict = model.batch_to_feed_dict(batch, is_train=False)
    
        prediction_generator = model.anytime_generator(feed_dict, len(model.prediction_ops))
        latencies: List[float] = []
        model_predictions: Dict[str, Any] = dict()

        start = time.time()
        for prediction_op, prediction in zip(model.prediction_ops, prediction_generator):
            model_predictions[prediction_op] = prediction
            latencies.append(time.time() - start)

        for batch_index in range(batch_size):
            prediction = None
            has_completed = False

            true_label = batch['output'][batch_index][0][0]

            for i, prediction_op in enumerate(model.prediction_ops):
                prediction = model_predictions[prediction_op][batch_index][0]

                accuracy_dict[prediction_op].append(1.0 - abs(prediction - true_label))
                latency_dict[prediction_op].append(latencies[i])

                if abs(prediction - 1.0) < SMALL_NUMBER:
                    precision_dict[prediction_op].append(float(true_label))

                if abs(true_label - 1.0) < SMALL_NUMBER:
                    recall_dict[prediction_op].append(float(prediction))

                # Prediction was zero, so short circuit the computation
                if not has_completed and (abs(prediction) < SMALL_NUMBER or i == len(model.prediction_ops) - 1):
                    computed_levels.append(i+1)
                    
                    accuracy_dict['scheduled_model'].append(1.0 - abs(prediction - true_label))
                    latency_dict['scheduled_model'].append(latencies[i])
                    
                    if abs(prediction - 1.0) < SMALL_NUMBER:
                        precision_dict['scheduled_model'].append(float(true_label))

                    if abs(true_label - 1.0) < SMALL_NUMBER:
                        recall_dict['scheduled_model'].append(float(prediction))

                    has_completed = True

        if max_num_batches is not None and batch_num >= max_num_batches:
            break

    stats_dict: DefaultDict[str, Dict[str, List[float]]] = defaultdict(dict)

    # Add levels
    stats_dict['levels']['scheduled_model'] = float(np.average(computed_levels))

    # Save average accuracy per model
    for model_name, accuracies in accuracy_dict.items():
        stats_dict['accuracy'][model_name] = accuracies

    # Save model latency
    for model_name, latencies in latency_dict.items():
        stats_dict['latency'][model_name] = latencies[1:]

    for model_name, precision in precision_dict.items():
        stats_dict['precision'][model_name] = precision

    for model_name, recall in recall_dict.items():
        stats_dict['recall'][model_name] = recall

    return stats_dict

def model_test(path: str, dataset_folder: str, max_num_batches: Optional[int]):
   
    save_folder, model_file = os.path.split(path)
    
    model_name = extract_model_name(model_file)
    if model_name.endswith('-loss'):
        model_name = model_name[:-len('-loss')]

    hypers_name = 'model-hyper-params-{0}.pkl.gz'.format(model_name)
    hyperparams_file = os.path.join(save_folder, hypers_name)
    hypers = extract_hyperparameters(hyperparams_file)[0]

    train_folder = os.path.join(dataset_folder, 'train')
    valid_folder = os.path.join(dataset_folder, 'valid')
    test_folder = os.path.join(dataset_folder, 'test')
    dataset = RNNSampleDataset(train_folder, valid_folder, test_folder)

    model = RNNSampleModel(hyper_parameters=hypers, save_folder=save_folder)

    # Build model
    model.restore_parameters(name=model_name)
    model.make(is_train=False)
    model.restore_weights(name=model_name)

    # Test the model
    stats_dict = test_accuracy(model, dataset, 1, max_num_batches)

    accuracy_file = RichPath.create(save_folder).join('model-accuracy-{0}.jsonl.gz'.format(model_name))
    accuracy_file.save_as_compressed_file([stats_dict])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--dataset-folder', type=str, required=True)
    parser.add_argument('--max-num-batches', type=int)
    args = parser.parse_args()

    model_test(args.model_path, args.dataset_folder, args.max_num_batches)
