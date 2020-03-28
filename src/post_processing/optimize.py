import os
import numpy as np
from argparse import ArgumentParser
from collections import namedtuple
from typing import Dict, Union, List, Optional

from models.adaptive_model import AdaptiveModel
from dataset.dataset import DataSeries
from dataset.rnn_sample_dataset import RNNSampleDataset
from utils.hyperparameters import HyperParameters
from utils.file_utils import extract_model_name, read_by_file_suffix, save_by_file_suffix
from utils.rnn_utils import get_logits_name
from utils.constants import HYPERS_PATH, TEST_LOG_PATH, TRAIN, VALID, TEST, METADATA_PATH, SMALL_NUMBER, BIG_NUMBER, OPTIMIZED_TEST_LOG_PATH
from utils.constants import SCHEDULED_OPTIMIZED, OPTIMIZED_RESULTS, OUTPUT
from utils.np_utils import thresholded_predictions, f1_score, precision, recall, sigmoid
from utils.testing_utils import ClassificationMetric
from utils.rnn_utils import get_prediction_name

from post_processing.threshold_optimizer_factory import get_optimizer


EvaluationResult = namedtuple('EvaluationResult', ['accuracy', 'precision', 'recall', 'f1_score', 'level', 'thresholds', 'latency', 'all_latency', 'flops'])


def print_eval_result(result: EvaluationResult):
    print(f'Results for thresholds: {result.thresholds}')
    print(f'Precision: {result.precision:.4f}')
    print(f'Recall: {result.recall:.4f}')
    print(f'F1 Score: {result.f1_score:.4f}')
    print(f'Accuracy: {result.accuracy:.4f}')
    print(f'Average Computed Levels: {result.level:.4f}')
    print(f'Average Latency: {result.latency:.4f}')
    print(f'Average Flops: {result.flops:.4f}')


def result_to_dict(result: EvaluationResult):
    return {key.upper(): value for key, value in result._asdict().items()}


def get_dataset(model_name: str, save_folder: str, dataset_folder: Optional[str]) -> RNNSampleDataset:
    metadata_file = os.path.join(save_folder, METADATA_PATH.format(model_name))
    metadata = read_by_file_suffix(metadata_file)

    if dataset_folder is None:
        train_folder = metadata['data_folders'][TRAIN.upper()]
        valid_folder = metadata['data_folders'][VALID.upper()]
        test_folder = metadata['data_folders'][TEST.upper()]
    else:
        assert os.path.exists(dataset_folder), f'The dataset folder {dataset_folder} does not exist!'
        train_folder = os.path.join(dataset_folder, TRAIN)
        valid_folder = os.path.join(dataset_folder, VALID)
        test_folder = os.path.join(dataset_folder, TEST)

    return RNNSampleDataset(train_folder, valid_folder, test_folder)


def get_model(model_name: str, hypers: HyperParameters, save_folder: str) -> AdaptiveModel:
    model = AdaptiveModel(hypers, save_folder, is_train=False)
    model.restore(name=model_name, is_train=False, is_frozen=False)
    return model


def evaluate_thresholds(model: AdaptiveModel,
                        thresholds: List[float],
                        dataset: RNNSampleDataset,
                        series: DataSeries,
                        test_log: Dict[str, Dict[str, float]]) -> EvaluationResult:
    test_dataset = dataset.minibatch_generator(series,
                                               metadata=model.metadata,
                                               batch_size=model.hypers.batch_size,
                                               should_shuffle=False,
                                               drop_incomplete_batches=True)

    logit_ops = [get_logits_name(i) for i in range(model.num_outputs)]

    predictions_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    levels_list: List[np.ndarray] = []
    latencies: List[float] = []
    flops: List[int] = []

    for batch_num, batch in enumerate(test_dataset):
        feed_dict = model.batch_to_feed_dict(batch, is_train=False)
        logits = model.execute(feed_dict, logit_ops)

        # Concatenate logits into a 2D array (logit_ops is already ordered by level)
        logits_concat = np.squeeze(np.concatenate([logits[op] for op in logit_ops], axis=-1))
        probabilities = sigmoid(logits_concat)
        labels = np.vstack(batch[OUTPUT])

        output = thresholded_predictions(probabilities, thresholds)
        predictions = output.predictions
        computed_levels = output.indices

        predictions_list.append(predictions)
        labels_list.append(labels)
        levels_list.append(computed_levels + 1.0)

        for level in computed_levels:
            level_name = get_prediction_name(level)
            latencies.append(test_log[level_name][ClassificationMetric.LATENCY.name])
            flops.append(test_log[level_name][ClassificationMetric.FLOPS.name])

        print(f'Completed batch {batch_num + 1}', end='\r')
    print()

    predictions = np.expand_dims(np.concatenate(predictions_list, axis=0), axis=-1)
    labels = np.vstack(labels_list)

    avg_levels = np.average(np.vstack(levels_list))
    p = precision(predictions, labels)
    r = recall(predictions, labels)
    f1 = f1_score(predictions, labels)
    accuracy = np.average(1.0 - np.abs(predictions - labels))
    avg_latency = np.average(latencies)
    avg_flops = np.average(flops)

    return EvaluationResult(precision=p,
                            recall=r,
                            f1_score=f1,
                            accuracy=accuracy,
                            level=avg_levels,
                            latency=avg_latency,
                            all_latency=latencies,
                            thresholds=list(thresholds),
                            flops=avg_flops)


def optimize_thresholds(optimizer_params: Dict[str, Union[float, int, str]], path: str, dataset_folder: Optional[str]):
    save_folder, model_file = os.path.split(path)

    model_name = extract_model_name(model_file)
    assert model_name is not None, f'Could not extract name from file: {model_file}'

    # Extract hyperparameters
    hypers_path = os.path.join(save_folder, HYPERS_PATH.format(model_name))
    hypers = HyperParameters.create_from_file(hypers_path)

    dataset = get_dataset(model_name, save_folder, dataset_folder)
    model = get_model(model_name, hypers, save_folder)

    # Get test log
    test_log_path = os.path.join(save_folder, TEST_LOG_PATH.format(model_name))
    assert os.path.exists(test_log_path), f'Must perform model testing before post processing'
    test_log = list(read_by_file_suffix(test_log_path))[0]

    print('Starting optimization')

    opt_outputs: List[OptimizerOutput] = []
    for _ in range(optimizer_params['instances']):
        optimizer = get_optimizer(name=optimizer_params['name'],
                                  iterations=optimizer_params['iterations'],
                                  batch_size=optimizer_params['batch_size'],
                                  level_weight=optimizer_params['level_weight'],
                                  **optimizer_params['opt_params'])
        output = optimizer.optimize(model, dataset)
        
        opt_outputs.append(output)
        print('==========')

    print('Completed optimization. Choosing the best model...')
    if len(opt_outputs) == 1:
        best_thresholds = opt_outputs[0].thresholds

        final_result = evaluate_thresholds(model=model,
                                           thresholds=best_thresholds,
                                           dataset=dataset,
                                           series=DataSeries.TEST,
                                           test_log=test_log)
        test_results = [final_result]
    else:
        best_thresholds, final_thresholds = None, None
        best_f1_score = -BIG_NUMBER
        final_result = None
        test_results = []

        for opt_output in opt_outputs:
            validation_result = evaluate_thresholds(model=model,
                                                    thresholds=opt_output.thresholds,
                                                    dataset=dataset,
                                                    series=DataSeries.VALID,
                                                    test_log=test_log)

            test_result = evaluate_thresholds(model=model,
                                              thresholds=opt_output.thresholds,
                                              dataset=dataset,
                                              series=DataSeries.TEST,
                                              test_log=test_log)
            test_results.append(test_result)

            # Evaluate models based on the validation set
            if validation_result.f1_score > best_f1_score:
                best_thresholds = validation_result.thresholds
                best_f1_score = validation_result.f1_score
                final_result = test_result

    print('Completed selection and optimization testing. Starting baseline evaluation....')

    baseline = [0.5 for _ in output.thresholds]
    result = evaluate_thresholds(model=model,
                                 thresholds=baseline,
                                 dataset=dataset,
                                 series=DataSeries.TEST,
                                 test_log=test_log)
    print_eval_result(result)

    print('===============')

    print_eval_result(final_result)

    # Save new results
    test_log[SCHEDULED_OPTIMIZED] = result_to_dict(final_result)
    test_log[OPTIMIZED_RESULTS] = list(map(result_to_dict, test_results))
    optimized_test_log_path = os.path.join(save_folder, OPTIMIZED_TEST_LOG_PATH.format(optimizer_params['name'], model_name))
    save_by_file_suffix([test_log], optimized_test_log_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--optimizer-params', type=str, required=True)
    parser.add_argument('--dataset-folder', type=str)
    args = parser.parse_args()

    optimizer_params_file = args.optimizer_params
    assert os.path.exists(optimizer_params_file), f'The file {optimizer_params_file} does not exist'

    optimizer_params = read_by_file_suffix(optimizer_params_file)

    optimize_thresholds(optimizer_params, args.model_path, args.dataset_folder)
