import os.path
import numpy as np
import math
from collections import namedtuple
from typing import Dict, Any, Tuple, List

from models.adaptive_model import AdaptiveModel
from models.standard_model import StandardModel, StandardModelType, SKIP_GATES
from dataset.dataset import Dataset, DataSeries
from utils.file_utils import save_by_file_suffix, read_by_file_suffix
from utils.rnn_utils import get_logits_name, get_stop_output_name
from utils.constants import OUTPUT, LOGITS, SEQ_LENGTH


POWER = np.array([24.085, 32.776, 37.897, 43.952, 48.833, 50.489, 54.710, 57.692, 59.212, 59.251])
ModelResults = namedtuple('ModelResults', ['predictions', 'labels', 'stop_probs', 'accuracy'])


def get_power_for_levels(power: np.ndarray, num_levels: int) -> np.ndarray:
    assert num_levels <= len(power), 'Must have fewer levels than power estimates'    

    if len(power) == num_levels:
        return power

    return power[:num_levels]


def clip(x: int, bounds: Tuple[int, int]) -> int:
    if x > bounds[1]:
        return bounds[1]
    elif x < bounds[0]:
        return bounds[0]
    return x


def save_test_log(accuracy: float, power: float, budget: float, noise_loc: float, output_file: str):
    test_log: Dict[float, Dict[str, Any]] = dict()
    if os.path.exists(output_file):
        test_log = list(read_by_file_suffix(output_file))[0]

    log_value = {
        'SHIFT': noise_loc,
        'ACCURACY': accuracy,
        'AVG_POWER': power,
        'BUDGET': budget
    }
    test_log['{0} {1}'.format(budget, noise_loc)] = log_value

    save_by_file_suffix([test_log], output_file)


def interpolate_power(power: np.ndarray, num_levels: int) -> List[float]:
    num_readings = len(power)
    assert int(math.ceil(num_levels / num_readings)) == int(num_levels / num_readings), 'Number of levels must be a multiple of the number of budgets'

    stride = int(num_levels / num_readings)
    power_readings: List[float] = []

    # For levels below the stride, we interpolate up to the first reading
    start = power[0] * 0.9
    end = power[0]
    interpolated_power = np.linspace(start=start, stop=end, endpoint=True, num=stride)
    power_readings.extend(interpolated_power[:-1])

    for i in range(1, len(power)):
        interpolated_power = np.linspace(start=power[i-1], stop=power[i], endpoint=False, num=stride)
        power_readings.extend(interpolated_power)

    # Add in the final reading
    power_readings.append(power[-1])

    return power_readings


def get_budget_index(power: np.ndarray, budget: int, level_accuracy: np.ndarray) -> int:
    num_levels = level_accuracy.shape[0]

    power_readings = interpolate_power(power=power, num_levels=num_levels)

    fixed_index = 0
    best_index = 0
    best_acc = 0.0
    while fixed_index < num_levels and power_readings[fixed_index] < budget:
        if best_acc < level_accuracy[fixed_index]:
            best_acc = level_accuracy[fixed_index]
            best_index = fixed_index

        fixed_index += 1

    return best_index


def execute_adaptive_model(model: AdaptiveModel, dataset: Dataset, series: DataSeries) -> ModelResults:
    """
    Executes the neural network on the given data series. We do this in a separate step
    to avoid recomputing for multiple budgets. Executing the neural network is relatively expensive.

    Args:
        model: The adaptive model used to perform inference
        dataset: The dataset to perform inference on
        series: The data series to extract. This is usually the TEST set.
    Returns:
        A model result tuple containing the inference results.
    """
    level_predictions: List[np.ndarray] = []
    stop_probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    num_outputs = model.num_outputs

    # Operations to execute
    logit_ops = [get_logits_name(i) for i in range(num_outputs)]
    stop_output_ops = [get_stop_output_name(i) for i in range(num_outputs)]

    # Make the batch generator. Don't shuffle so we have consistent results.
    data_generator = dataset.minibatch_generator(series=series,
                                                 batch_size=model.hypers.batch_size,
                                                 metadata=model.metadata,
                                                 should_shuffle=False)

    for batch_num, batch in enumerate(data_generator):
        # Compute the predicted log probabilities
        feed_dict = model.batch_to_feed_dict(batch, is_train=False, epoch_num=0)
        model_results = model.execute(feed_dict, logit_ops + stop_output_ops)

        # Concatenate logits into a [B, L, C] array (logit_ops is already ordered by level).
        # For reference, L is the number of levels and C is the number of classes
        logits_concat = np.concatenate([np.expand_dims(model_results[op], axis=1) for op in logit_ops], axis=1)

        # Concatenate stop outputs into a [B, L] array if supported by this model
        stop_outputs = [np.expand_dims(model_results[op], axis=1) for op in stop_output_ops if op in model_results]
        stop_output_concat = np.concatenate(stop_outputs, axis=1)
        stop_probs.append(stop_output_concat) 

        # Compute the predictions for each level
        level_pred = np.argmax(logits_concat, axis=-1)  # [B, L]
        level_predictions.append(level_pred)

        labels.append(np.array(batch[OUTPUT]).reshape(-1, 1))

    # Save results as attributes
    level_predictions = np.concatenate(level_predictions, axis=0)
    labels = np.concatenate(labels, axis=0)  # [N, L]
    stop_probs = np.concatenate(stop_probs, axis=0)
    level_accuracy = np.average((level_predictions == labels.astype).astype(float), axis=0)

    return ModelResults(predictions=level_predictions, labels=labels, stop_probs=stop_probs, accuracy=level_accuracy)


def execute_standard_model(model: StandardModel, dataset: Dataset, series: DataSeries) -> ModelResults:
    """
    Executes the neural network on the given data series. We do this in a separate step
    to avoid recomputing for multiple budgets. Executing the neural network is relatively expensive.

    Args:
        model: The standard model used to perform inference
        dataset: The dataset to perform inference on
        series: The data series to extract. This is usually the TEST set.
    Returns:
        A model result tuple containing the inference results.
    """
    level_predictions: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    # Make the batch generator. Don't shuffle so we have consistent results.
    data_generator = dataset.minibatch_generator(series=series,
                                                 batch_size=model.hypers.batch_size,
                                                 metadata=model.metadata,
                                                 should_shuffle=False)

    for batch_num, batch in enumerate(data_generator):
        # Compute the predicted log probabilities
        feed_dict = model.batch_to_feed_dict(batch, is_train=False, epoch_num=0)
        model_results = model.execute(feed_dict, [LOGITS])

        # Compute the predictions for each level
        level_pred = np.argmax(model_results[LOGITS], axis=-1)  # [B, L]
        level_predictions.append(level_pred)

        labels.append(np.array(batch[OUTPUT]).reshape(-1, 1))

    # Save results as attributes
    level_predictions = np.concatenate(level_predictions, axis=0)
    labels = np.concatenate(labels, axis=0)  # [N, L]
    level_accuracy = np.average((level_predictions == labels).astype(float), axis=0)

    return ModelResults(predictions=level_predictions, labels=labels, stop_probs=None, accuracy=level_accuracy)


def execute_skip_rnn_model(model: StandardModel, dataset: Dataset, series: DataSeries) -> ModelResults:
    """
    Executes the neural network on the given data series. We do this in a separate step
    to avoid recomputing for multiple budgets. Executing the neural network is relatively expensive.

    Args:
        model: The Skip RNN standard model used to perform inference
        dataset: The dataset to perform inference on
        series: The data series to extract. This is usually the TEST set.
    Returns:
        A model result tuple containing the inference results. The sample fractions are placed in the stop_probs element.
    """
    assert model.model_type == StandardModelType.SKIP_RNN, 'Must provide a Skip RNN'
    seq_length = model.metadata[SEQ_LENGTH]

    predictions: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    sample_counts = np.zeros(shape=(seq_length, ), dtype=np.int64)

    # Make the batch generator. Don't shuffle so we have consistent results.
    data_generator = dataset.minibatch_generator(series=series,
                                                 batch_size=64,
                                                 metadata=model.metadata,
                                                 should_shuffle=False)

    for batch_num, batch in enumerate(data_generator):
        # Compute the predicted log probabilities
        feed_dict = model.batch_to_feed_dict(batch, is_train=False, epoch_num=0)
        model_results = model.execute(feed_dict, [LOGITS, SKIP_GATES])

        # Compute the predictions for each level
        pred = np.argmax(model_results[LOGITS], axis=-1)  # [B]
        predictions.append(pred.reshape(-1, 1))

        # Collect the number of samples processed for each batch element. We subtract 1
        # because it is impossible for the models to consume zero samples.
        num_samples = np.sum(model_results[SKIP_GATES], axis=-1).astype(int) - 1  # [B]
        counts = np.bincount(num_samples, minlength=seq_length)
        sample_counts += counts  # [T]

        labels.append(np.array(batch[OUTPUT]).reshape(-1, 1))

    # Save results as attributes
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)  # [N, 1]
    accuracy = np.average((predictions == labels).astype(float), axis=0)

    # Normalize the sample counts
    sample_counts = sample_counts.astype(float)
    sample_fractions = sample_counts / np.sum(sample_counts)

    return ModelResults(predictions=predictions, labels=labels, stop_probs=sample_fractions, accuracy=accuracy)
