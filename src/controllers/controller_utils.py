import os.path
import numpy as np
import math
from collections import namedtuple
from typing import Dict, Any, Tuple, List, Optional

from models.adaptive_model import AdaptiveModel
from models.standard_model import StandardModel
from dataset.dataset import Dataset, DataSeries
from utils.file_utils import save_by_file_suffix, read_by_file_suffix
from utils.sequence_model_utils import SequenceModelType
from utils.constants import OUTPUT, LOGITS, SEQ_LENGTH, SKIP_GATES, PHASE_GATES, STOP_OUTPUT_NAME
from controllers.power_utils import get_avg_power


ModelResults = namedtuple('ModelResults', ['predictions', 'labels', 'stop_probs', 'accuracy'])
BATCH_SIZE = 64


def clip(x: int, bounds: Tuple[int, int]) -> int:
    if x > bounds[1]:
        return bounds[1]
    elif x < bounds[0]:
        return bounds[0]
    return x


def save_test_log(accuracy: float, power: float, valid_accuracy: Optional[float], budget: float, key: str, output_file: str):
    test_log: Dict[str, Dict[str, Any]] = dict()
    if os.path.exists(output_file):
        test_log = list(read_by_file_suffix(output_file))[0]

    if key not in test_log:
        test_log[key] = dict()

    log_value = {
        'ACCURACY': accuracy,
        'AVG_POWER': power,
        'VALID_ACCURACY': valid_accuracy,
        'BUDGET': budget
    }
    budget_str = '{0:.4f}'.format(budget)
    test_log[key][budget_str] = log_value

    save_by_file_suffix([test_log], output_file)


def get_budget_index(budget: float, valid_accuracy: np.ndarray, max_time: int, power_estimates: np.ndarray, allow_violations: bool) -> int:
    """
    Selects the single model level which should yield the best overall accuracy. This decision
    is based on the validation accuracy for each level.

    Args:
        budget: The current avg power budget
        valid_accuracy: A [L] array containing the validation accuracy for each model level
        max_time: The number of timesteps
        power_estimates: A [L] array of power estimates for each level
        allow_violations: Index selected in a manner which allows for budget violations if such violations
            will lead to better end-to-end accuracy.
    Returns:
        The "optimal" model level.
    """
    best_index = 0
    best_acc = 0.0

    if allow_violations:
        num_levels = valid_accuracy.shape[0]
        energy_budget = budget * max_time

        for level_idx in range(num_levels):
            # Estimate the number of timesteps on which we can perform inference with this level
            avg_power = power_estimates[level_idx]
            projected_timesteps = min(energy_budget / avg_power, max_time)

            projected_correct = valid_accuracy[level_idx] * projected_timesteps
            estimated_accuracy = projected_correct / max_time

            if estimated_accuracy > best_acc:
                best_acc = estimated_accuracy
                best_index = level_idx
    else:
        budget_comparison = power_estimates <= budget
        if np.any(budget_comparison):
            budget_mask = budget_comparison.astype(float)
            masked_accuracy = valid_accuracy * budget_mask
            best_index = np.argmax(masked_accuracy)
        else:
            best_index = np.argmin(power_estimates)

    return best_index


def concat_model_results(model_results: List[ModelResults]) -> ModelResults:
    """
    Stacks each field of the given results into a single array. This is useful
    for Skip RNN and Phased RNN systems in which each output is a separate model.
    """
    predictions = np.concatenate([r.predictions for r in model_results], axis=1)  # [N, L]
    labels = model_results[0].labels  # [N, 1]
    stop_probs = [r.stop_probs for r in model_results]
    accuracy = [r.accuracy for r in model_results]
    return ModelResults(predictions=predictions,
                        labels=labels,
                        stop_probs=stop_probs,
                        accuracy=accuracy)


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
    ops = [LOGITS, STOP_OUTPUT_NAME]

    # Make the batch generator. Don't shuffle so we have consistent results.
    data_generator = dataset.minibatch_generator(series=series,
                                                 batch_size=BATCH_SIZE,
                                                 metadata=model.metadata,
                                                 should_shuffle=False)

    for batch_num, batch in enumerate(data_generator):
        # Compute the predicted log probabilities
        feed_dict = model.batch_to_feed_dict(batch, is_train=False, epoch_num=0)
        model_results = model.execute(feed_dict, ops=ops)

        # Concatenate logits into a [B, L, C] array (logit_ops is already ordered by level).
        # For reference, L is the number of levels and C is the number of classes
        logits = model_results[LOGITS]

        stop_output = model_results[STOP_OUTPUT_NAME]  # [B, L]
        stop_probs.append(stop_output)

        # Compute the predictions for each level
        level_pred = np.argmax(logits, axis=-1)  # [B, L]
        level_predictions.append(level_pred)

        labels.append(np.array(batch[OUTPUT]).reshape(-1, 1))

    # Save results as attributes
    level_predictions = np.concatenate(level_predictions, axis=0)
    labels = np.concatenate(labels, axis=0)  # [N, 1]
    stop_probs = np.concatenate(stop_probs, axis=0)
    level_accuracy = np.average((level_predictions == labels).astype(float), axis=0)

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
                                                 batch_size=BATCH_SIZE,
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
    labels = np.concatenate(labels, axis=0)  # [N, 1]

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
    assert model.model_type == SequenceModelType.SKIP_RNN, 'Must provide a Skip RNN'
    seq_length = model.seq_length

    predictions: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    sample_counts = np.zeros(shape=(seq_length, ), dtype=np.int64)

    # Make the batch generator. Don't shuffle so we have consistent results.
    data_generator = dataset.minibatch_generator(series=series,
                                                 batch_size=BATCH_SIZE,
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


def execute_phased_rnn_model(model: StandardModel, dataset: Dataset, series: DataSeries) -> ModelResults:
    """
    Executes the neural network on the given data series. We do this in a separate step
    to avoid recomputing for multiple budgets. Executing the neural network is relatively expensive.

    Args:
        model: The Phased RNN standard model used to perform inference
        dataset: The dataset to perform inference on
        series: The data series to extract. This is usually the TEST set.
    Returns:
        A model result tuple containing the inference results. The sample fractions are placed in the stop_probs element.
    """
    assert model.model_type == SequenceModelType.PHASED_RNN, 'Must provide a Phased RNN'
    seq_length = model.seq_length

    predictions: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    sample_counts = np.zeros(shape=(seq_length, ), dtype=np.int64)

    # Make the batch generator. Don't shuffle so we have consistent results.
    data_generator = dataset.minibatch_generator(series=series,
                                                 batch_size=BATCH_SIZE,
                                                 metadata=model.metadata,
                                                 should_shuffle=False)

    for batch_num, batch in enumerate(data_generator):
        # Compute the predicted log probabilities
        feed_dict = model.batch_to_feed_dict(batch, is_train=False, epoch_num=0)
        model_results = model.execute(feed_dict, [LOGITS, PHASE_GATES])

        # Compute the predictions for each level
        pred = np.argmax(model_results[LOGITS], axis=-1)  # [B]
        predictions.append(pred.reshape(-1, 1))

        # Collect the number of samples processed for each batch element. We subtract 1
        # because it is impossible for the models to consume zero samples.
        phase_gates = model_results[PHASE_GATES]  # [B, T]
        num_samples = np.count_nonzero(phase_gates, axis=-1) - 1  # [B]
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
