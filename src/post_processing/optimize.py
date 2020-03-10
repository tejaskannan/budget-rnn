import os
import numpy as np
from argparse import ArgumentParser
from dpu_utils.utils import RichPath
from typing import Dict, Union, List

from models.rnn_model import RNNModel
from dataset.dataset import DataSeries
from dataset.rnn_sample_dataset import RNNSampleDataset
from utils.hyperparameters import HyperParameters
from utils.file_utils import extract_model_name
from utils.rnn_utils import get_logits_name
from utils.constants import HYPERS_PATH, TEST_LOG_PATH, TRAIN, VALID, TEST, METADATA_PATH, SMALL_NUMBER
from utils.misc import sigmoid
from utils.np_utils import thresholded_predictions, f1_score, precision, recall
from post_processing.threshold_optimizer import ThresholdOptimizer



def get_dataset(model_name: str, save_folder: RichPath) -> RNNSampleDataset:
    metadata_file = save_folder.join(METADATA_PATH.format(model_name))
    metadata = metadata_file.read_by_file_suffix()
    
    # TODO: Fix this hack
    train_folder = os.path.join('../', metadata['data_folders'][TRAIN.upper()].path)
    valid_folder = os.path.join('../', metadata['data_folders'][VALID.upper()].path)
    test_folder = os.path.join('../', metadata['data_folders'][TEST.upper()].path)

    return RNNSampleDataset(train_folder, valid_folder, test_folder)


def get_model(model_name: str, hypers: HyperParameters, save_folder: RichPath) -> RNNModel:
    model = RNNModel(hypers, save_folder)
    model.restore(name=model_name, is_train=False)
    return model


def evaluate_thresholds(model: RNNModel, dataset: RNNSampleDataset, thresholds: List[float]):
    test_dataset = dataset.minibatch_generator(DataSeries.TEST,
                                               metadata=model.metadata,
                                               batch_size=model.hypers.batch_size,
                                               should_shuffle=False,
                                               drop_incomplete_batches=True)

    logit_ops = [get_logits_name(i) for i in range(model.num_outputs)]

    predictions_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    levels_list: List[np.ndarray] = []

    for batch_num, batch in enumerate(test_dataset):
        feed_dict = model.batch_to_feed_dict(batch, is_train=False)
        logits = model.execute(feed_dict, logit_ops)

        # Concatenate logits into a 2D array (logit_ops is already ordered by level)
        logits_concat = np.squeeze(np.concatenate([logits[op] for op in logit_ops], axis=-1))
        probabilities = sigmoid(logits_concat)
        labels = np.vstack(batch['output'])

        output = thresholded_predictions(probabilities, thresholds)
        predictions = output.predictions
        computed_levels = output.indices

        predictions_list.append(predictions)
        labels_list.append(labels)
        levels_list.append(computed_levels)

        print(f'Completed testing batch {batch_num + 1}', end='\r')
    print()

    avg_levels = np.average(np.vstack(levels_list))

    predictions = np.expand_dims(np.concatenate(predictions_list, axis=0), axis=-1)
    labels = np.vstack(labels_list)

    p = precision(predictions, labels)
    r = recall(predictions, labels)
    f1 = f1_score(predictions, labels)

    print(f'Results for thresholds: {thresholds}')
    print(f'Precision: {p:.4f}')
    print(f'Recall: {r:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Average Computed Levels: {avg_levels:.4f}')

def optimize_thresholds(optimizer_params: Dict[str, Union[float, int]], path: str):
    save_folder, model_file = os.path.split(path)

    model_name = extract_model_name(model_file)
    assert model_name is not None, f'Could not extract name from file: {model_file}'

    save_folder = RichPath.create(save_folder)

    # Extract hyperparameters
    hypers_name = HYPERS_PATH.format(model_name)
    hypers = HyperParameters.create_from_file(save_folder.join(hypers_name))

    dataset = get_dataset(model_name, save_folder)
    model = get_model(model_name, hypers, save_folder)

    optimizer = ThresholdOptimizer(population_size=optimizer_params['population_size'],
                                   mutation_rate=optimizer_params['mutation_rate'],
                                   batch_size=optimizer_params['batch_size'],
                                   selection_count=optimizer_params['selection_count'],
                                   iterations=optimizer_params['iterations'])

    output = optimizer.optimize(model, dataset)

    print('Completed optimization. Starting evaluation.')

    baseline = [0.5 for _ in output.thresholds]
    evaluate_thresholds(model, dataset, baseline)
    print('===============')
    evaluate_thresholds(model, dataset, output.thresholds)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--optimizer-params', type=str, required=True)
    args = parser.parse_args()

    optimizer_params_file = RichPath.create(args.optimizer_params)
    assert optimizer_params_file.exists(), f'The file {optimizer_params_file} does not exist'

    optimizer_params = optimizer_params_file.read_by_file_suffix()

    optimize_thresholds(optimizer_params, args.model_path)
