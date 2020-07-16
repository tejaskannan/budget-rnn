import numpy as np
import matplotlib.pyplot as plt
import math
import os.path
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Tuple, List, Union, Optional, Dict

from models.base_model import Model
from models.model_factory import get_model
from models.adaptive_model import AdaptiveModel
from models.standard_model import StandardModel
from dataset.dataset import DataSeries, Dataset
from dataset.dataset_factory import get_dataset
from utils.hyperparameters import HyperParameters
from utils.rnn_utils import get_logits_name, get_states_name, AdaptiveModelType, is_cascade
from utils.file_utils import extract_model_name, read_by_file_suffix, save_by_file_suffix, make_dir
from utils.np_utils import min_max_normalize, round_to_precision
from utils.constants import OUTPUT, SMALL_NUMBER, INPUTS, OPTIMIZED_TEST_LOG_PATH, METADATA_PATH, HYPERS_PATH, PREDICTION, SEQ_LENGTH
from utils.adaptive_inference import normalize_logits, threshold_predictions
from utils.testing_utils import ClassificationMetric
from controllers.logistic_regression_controller import Controller, CONTROLLER_PATH, get_power_for_levels, POWER, RandomController


PowerEstimate = namedtuple('PowerEstimate', ['avg_power', 'fraction'])
SMOOTHING_FACTOR = 100


def make_dataset(model_name: str, save_folder: str, dataset_type: str, dataset_folder: Optional[str]) -> Dataset:
    metadata_file = os.path.join(save_folder, METADATA_PATH.format(model_name))
    metadata = read_by_file_suffix(metadata_file)

    # Infer the dataset
    if dataset_folder is None:
        dataset_folder = os.path.dirname(metadata['data_folders'][TRAIN.upper()])

    # Validate the dataset folder
    assert os.path.exists(dataset_folder), f'The dataset folder {dataset_folder} does not exist!'

    return get_dataset(dataset_type=dataset_type, data_folder=dataset_folder)


def make_model(model_name: str, hypers: HyperParameters, save_folder: str) -> Model:
    model = get_model(hypers, save_folder, is_train=False)
    model.restore(name=model_name, is_train=False, is_frozen=False)
    return model


def get_serialized_info(model_path: str, dataset_folder: Optional[str]) -> Tuple[AdaptiveModel, Dataset]:
    save_folder, model_file = os.path.split(model_path)

    model_name = extract_model_name(model_file)
    assert model_name is not None, f'Could not extract name from file: {model_file}'

    # Extract hyperparameters
    hypers_path = os.path.join(save_folder, HYPERS_PATH.format(model_name))
    hypers = HyperParameters.create_from_file(hypers_path)

    dataset = make_dataset(model_name, save_folder, hypers.dataset_type, dataset_folder)
    model = make_model(model_name, hypers, save_folder)

    return model, dataset


def get_baseline_model_results(model: Model, dataset: Dataset, series: DataSeries, shuffle: bool) -> Tuple[np.ndarray, np.ndarray]:
    labels: List[np.ndarray] = []
    predictions: List[np.ndarray] = []

    data_generator = dataset.minibatch_generator(series=series,
                                                 metadata=model.metadata,
                                                 batch_size=model.hypers.batch_size,
                                                 should_shuffle=shuffle)
    for batch in data_generator:
        feed_dict = model.batch_to_feed_dict(batch, is_train=False, epoch_num=0)

        batch_predictions = model.execute(ops=[PREDICTION], feed_dict=feed_dict)

        predictions.append(batch_predictions[PREDICTION])
        labels.append(np.array(batch[OUTPUT]).reshape(-1, 1))

    predictions = np.vstack(predictions)
    labels = np.vstack(labels)

    return labels, predictions


def clip(x: int, bounds: Tuple[int, int]) -> int:
    if x > bounds[1]:
        return bounds[1]
    elif x < bounds[0]:
        return bounds[0]
    return x


class PIController:

    def __init__(self, kp: float, ki: float):
        self._kp = kp
        self._ki = ki

        self._errors: List[float] = []
        self._times: List[float] = []
        self._integral = 0.0

    def errors(self) -> List[float]:
        return self._errors

    def times(self) -> List[float]:
        return self._times

    def plant_function(self, y_pred: Union[float, int], proportional_error: float, integral_error: float) -> Union[float, int]:
        raise NotImplementedError()

    def update(self, **kwargs):
        """
        A general update function to update any controller-specific parameters
        """
        pass

    def step(self, y_true: Union[float, int], y_pred: Union[float, int], time: float) -> Union[float, int]:
        """
        Updates the controller and outputs the next control signal.
        """
        error = float(y_true - y_pred)

        if len(self._errors) > 0:
            h = (error + self._errors[-1]) / 2
            w = time - self._times[-1]
            self._integral += h * w

        integral_error = self._ki * self._integral
        proportional_error = self._kp * error

        self._errors.append(error)
        self._times.append(time)

        return self.plant_function(y_true, y_pred, proportional_error, integral_error)

    def reset(self):
        """
        Resets the PI Controller.
        """
        self._errors = []
        self._times = []
        self._integral = 0.0


class BudgetController(PIController):

    def __init__(self, kp: float, ki: float, output_range: Tuple[int, int], budget: float, margin: float, power_factor: float):
        super().__init__(kp, ki)
        self._output_range = output_range
        self._budget = budget
        self._margin = margin
        self._power_factor = power_factor

    def update(self, **kwargs):
        anneal_rate = kwargs['anneal_rate']
        self._margin = self._margin * anneal_rate

    def plant_function(self, y_true: Union[float, int], y_pred: Union[float, int], proportional_error: float, integral_error: float) -> Union[float, int]:
        # Error in power budget. For now, we only care about the upper bound budget
        control_signal = proportional_error + integral_error

        power = y_pred
        step = abs(control_signal) * self._power_factor

        lower_limit = y_true * (1.0 - self._margin)
        upper_limit = y_true * (1.0 + self._margin)

        if power <= upper_limit:
            return 0  # By returning the highest # of levels, we allow the model controller to control
        
        sign = 1
        if power > upper_limit:
            sign = -1

        step = int(math.floor(sign * step))

        if step == 0:
            return sign
        return step


class BudgetDistribution:

    def __init__(self,
                 prior_counts: Dict[int, np.ndarray],
                 budget: float,
                 max_time: int,
                 num_levels: int,
                 num_classes: int,
                 warmup_frac: float,
                 panic_frac: float,
                 margin: float,
                 anneal_rate: float,
                 power: np.ndarray):
        self._prior_counts = prior_counts  # key: class index, value: array [L] counts for each level
        self._max_time = max_time
        self._budget = budget
        self._num_levels = num_levels
        self._num_classes = num_classes
        self._warmup_time = int(warmup_frac * max_time)
        self._panic_time = int(panic_frac * max_time)
        self._margin = margin
        self._anneal_rate = anneal_rate
        self._power = power

        # Estimated count of each label over the time window
        self._estimated_label_counts = np.zeros(shape=(num_classes, ))
        total_count = sum(np.sum(counts) for counts in prior_counts.values())
        for label, counts in prior_counts.items():
            normalized_count = (np.sum(counts) + SMOOTHING_FACTOR) / (total_count + self._num_classes * SMOOTHING_FACTOR)
            self._estimated_label_counts[label] += normalized_count * max_time

        self._observed_label_counts = np.zeros_like(self._estimated_label_counts)

    def get_budget(self, time: int) -> float:
        if time < self._warmup_time:
            return budget * (1 + self._margin)
        elif time > self._panic_time:
            return budget

        expected_rest = 0
        variance_rest = 0
        time_delta = self._max_time - time

        class_count_diff = np.maximum(self._estimated_label_counts - self._observed_label_counts, 0)
        estimated_remaining = np.sum(class_count_diff)

        # We compute the MLE estimates for the mean and variance power given the observed samples
        # and the training set

        count_rest = 0
        for class_idx in range(self._num_classes):
            class_level_counts = self._prior_counts[class_idx]
            n_class = max(np.sum(class_level_counts), SMALL_NUMBER)
           
            # MLE estimate of the mean power
            power_mean = np.sum((class_level_counts * self._power) / (n_class))

            squared_diff = np.square(self._power - power_mean)
            power_var = np.sum((class_level_counts * squared_diff) / n_class)

            count_diff = max(self._estimated_label_counts[class_idx] - self._observed_label_counts[class_idx], 0)
            remaining_fraction = count_diff / estimated_remaining

            expected_rest += power_mean * remaining_fraction

            variance_rest += np.square(count_diff / time) * power_var

        expected_power = (1.0 / time) * (self._max_time * self._budget - time_delta * expected_rest)
        expected_power = max(expected_power, self._power[0])  # We clip the power to the lowest level (this is the lowest feasible amount)

        estimator_variance = 2 * (1.0 / time) * variance_rest
        current_budget = expected_power + estimator_variance

        return current_budget

    def update(self, label: int, levels: int, time: int):
        self._observed_label_counts[label] += 1
        self._prior_counts[label][levels] += 1

        if time > self._warmup_time:
            self._margin = self._margin * self._anneal_rate


# Function to get model results
def get_model_results(model: AdaptiveModel, dataset: Dataset, shuffle: bool, series: DataSeries):
    labels: List[np.ndarray] = []
    level_predictions: List[np.ndarray] = []
    states: List[np.ndarray] = []
    dataset_inputs: List[np.ndarray] = []

    seq_length = model.metadata[SEQ_LENGTH]
    num_sequences = model.num_sequences

    logit_ops = [get_logits_name(i) for i in range(model.num_outputs)]
    state_ops = [get_states_name(i) for i in range(model.num_outputs)]
    data_generator = dataset.minibatch_generator(series=series,
                                                 batch_size=model.hypers.batch_size,
                                                 metadata=model.metadata,
                                                 should_shuffle=shuffle)

    states_idx = 0
    if is_cascade(model.model_type):
        states_idx = -1

    for batch_num, batch in enumerate(data_generator):
        # Compute the predicted log probabilities
        feed_dict = model.batch_to_feed_dict(batch, is_train=False, epoch_num=0)
        model_results = model.execute(feed_dict, logit_ops + state_ops)

        first_states = np.concatenate([np.expand_dims(np.squeeze(model_results[op][states_idx]), axis=1) for op in state_ops], axis=1)

        inputs = np.array(batch[INPUTS])
        if is_cascade(model.model_type):
            stride = int(seq_length / num_sequences)
            first_inputs = np.concatenate([inputs[:, :, i, :] for i in range(0, seq_length, stride)], axis=1)
        else:
            first_inputs = np.concatenate([inputs[:, :, i, :] for i in range(num_sequences)], axis=1)

        state_features = np.concatenate([first_states, first_inputs], axis=-1)  # [B, D + K]
        states.append(state_features)

        dataset_inputs.append(np.squeeze(inputs, axis=1))

        # Concatenate logits into a [B, L, C] array (logit_ops is already ordered by level).
        # For reference, L is the number of levels and C is the number of classes
        logits_concat = np.concatenate([np.expand_dims(model_results[op], axis=1) for op in logit_ops], axis=1)

        # Compute the predictions for each level
        level_pred = np.argmax(logits_concat, axis=-1)  # [B, L]
        level_predictions.append(level_pred)

        # Normalize logits and round to fixed point representation
        true_values = np.squeeze(batch[OUTPUT])
        labels.append(true_values)

    labels = np.concatenate(labels, axis=0)
    level_predictions = np.concatenate(level_predictions, axis=0)
    states = np.concatenate(states, axis=0)
    dataset_inputs = np.concatenate(dataset_inputs, axis=0)

    level_acc = np.equal(level_predictions, np.expand_dims(labels, axis=-1)).astype(float)
    level_acc = np.average(level_acc, axis=0)

    return labels, level_predictions, states, level_acc, dataset_inputs


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


def get_accuracy_index(system_acc: float, level_accuracy: np.ndarray) -> int:
    num_levels = len(level_accuracy)
    diff = np.abs(system_acc - level_accuracy)
    nearest_level = np.argmin(diff)

    best_level_above = level_accuracy.shape[0] - 1
    for idx, acc in enumerate(level_accuracy):
        if acc >= system_acc:
            best_level_above = idx
            break

    return min(nearest_level, best_level_above)


def run_fixed_policy(labels: np.ndarray,
                     level_predictions: int,
                     policy_index: int,
                     power_estimates: np.ndarray,
                     noise: float) -> Tuple[List[float], List[float]]:
    rand = np.random.RandomState(seed=42)

    correct: List[float] = []
    power: List[float] = []

    power_readings = interpolate_power(power_estimates, level_predictions.shape[1])

    max_time = labels.shape[0]
    for t in range(max_time):
        label = int(labels[t])
        pred = int(level_predictions[t, policy_index])
        correct.append(float(label == pred))

        p = power_readings[policy_index] + rand.uniform(low=-noise, high=noise)
        power.append(p)

    return correct, power


def estimate_label_counts(controller: Controller, budget: float, num_levels: int, num_classes: int) -> Dict[int, np.ndarray]:
    levels, predictions = controller.predict_levels(series=DataSeries.VALID, budget=budget) 
    batch_size = predictions.shape[0]

    result: Dict[int, np.ndarray] = dict()

    # Estimate power for each class
    for class_idx in range(num_classes):
        class_mask = (predictions == class_idx).astype(int)

        class_level_counts = np.bincount(levels, minlength=num_levels, weights=class_mask)
        result[class_idx] = class_level_counts

    return result


def run_simulation(labels: np.ndarray,
                   level_predictions: np.ndarray,
                   states: np.ndarray,
                   power_estimates: np.ndarray,
                   budget: int,
                   controller_type: str,
                   num_levels: int,
                   num_classes: int,
                   noise: float,
                   precision: int,
                   model_path: str,
                   dataset_folder: str):
    # Create the model controller
    if controller_type == 'random':
        controller = RandomController(model_path=model_path, dataset_folder=dataset_folder, budgets=[budget], power=power_estimates)
        controller.fit(series=DataSeries.VALID)
    elif controller_type == 'logistic':
        save_folder, model_file_name = os.path.split(model_path)
        model_name = extract_model_name(model_file_name)
        controller = Controller.load(os.path.join(save_folder, CONTROLLER_PATH.format(model_name)))
    else:
        raise ValueError('Unknown controller name: {0}'.format(controller_type))

    # Execute model on the validation set and collect levels
    level_accuracy = np.average((level_predictions == np.expand_dims(labels, axis=1)).astype(float), axis=0)

    # Create a list of times
    max_time = level_predictions.shape[0]
    times = list(range(max_time))

    # Lists to save results
    power: List[float] = []
    errors: List[float] = []
    num_correct: List[float] = []
    predictions: List[float] = []
    desired_levels: List[int] = []

    levels_per_label: DefaultDict[int, List[int]] = defaultdict(list)

    rand = np.random.RandomState(seed=42)

    # Parameters for model schedule
    start_margin = 5.0
    end_margin = margin
    anneal_rate = np.exp((1.0 / max_time) * np.log((end_margin + SMALL_NUMBER) / start_margin))

    # Create the budget controller
    output_range = (0, num_levels - 1)
    budget_controller = BudgetController(kp=1.0, ki=0.0625, output_range=output_range, budget=budget, margin=margin, power_factor=1.0)
    current_budget = budget

    level_counts = np.zeros(shape=(num_levels, ))
    power_noise = rand.uniform(low=-noise, high=noise, size=(max_time, ))
    level_idx = np.arange(num_levels)

    # Create the power distribution
    prior_counts = estimate_label_counts(controller, budget, num_levels, num_classes)
    budget_distribution = BudgetDistribution(prior_counts=prior_counts,
                                             budget=budget,
                                             max_time=max_time,
                                             num_levels=num_levels,
                                             num_classes=num_classes,
                                             warmup_frac=0.0,
                                             panic_frac=0.95,
                                             margin=0.2,
                                             anneal_rate=0.9,
                                             power=power_estimates)

    for t in range(max_time):
        current_budget = budget_distribution.get_budget(t+1)

        # Use the control model to determine the number of states to collect
        y_pred_model = controller.predict_sample(inputs=states[t], budget=budget)

        # Make adjustments based on observed power
        avg_power = np.average(power) if len(power) > 0 else 0
        avg_power = np.clip(avg_power, a_min=current_budget, a_max=None)  # Clip to prevent negative error

        budget_step = budget_controller.step(y_true=current_budget, y_pred=avg_power, time=t)

        # Form predicted levels using both controllers
        y_pred = clip(y_pred_model + budget_step, bounds=output_range)

        # Compute the (noisy) power consumption for using this number of levels
        p = power_estimates[y_pred] + power_noise[t]

        power.append(p)
        errors.append(y_pred_model - y_pred)
        desired_levels.append(y_pred_model)

        model_prediction = level_predictions[t, y_pred]
        predictions.append(float(model_prediction))
        num_correct.append(float(model_prediction == labels[t]))

        levels_per_label[model_prediction].append(y_pred)
        level_counts[y_pred_model] += 1

        budget_distribution.update(label=model_prediction, levels=y_pred, time=t+1)

    print('Level Distribution: {0}'.format(level_counts / max_time))
    print('Accuracy: {0}'.format(np.average(num_correct)))

    # Print out the label distributions
    for label, label_levels in sorted(levels_per_label.items()):
        print('Label {0}: Avg Levels -> {1:.5f}, Std Levels -> {2:.5f}'.format(label, np.average(label_levels), np.std(label_levels)))

    return power, errors, num_correct, predictions, desired_levels

def plot_and_save(power: List[float],
                  errors: List[float],
                  num_correct: List[float],
                  predictions: List[float],
                  labels: List[float],
                  desired_levels: List[int],
                  model_path: str,
                  output_folder: Optional[str],
                  controller_type: str,
                  budget: int,
                  num_levels: int,
                  level_accuracy: np.ndarray,
                  level_predictions: np.ndarray,
                  noise: float,
                  baseline_labels: np.ndarray,
                  baseline_predictions: np.ndarray,
                  power_estimates: np.ndarray):
    times = np.arange(start=0, stop=len(power), dtype=float)
    avg_power = np.cumsum(power) / (times + 1)
    cumulative_accuracy = np.cumsum(num_correct) / (times + 1)

    # Create optimize test log path
    save_folder, model_file_name = os.path.split(model_path)
    model_name = extract_model_name(model_file_name)
    opt_log_file = os.path.join(save_folder, OPTIMIZED_TEST_LOG_PATH.format(controller_type, 'power', budget, model_name))
    print(opt_log_file)

    # Get statistics from simulation
    accuracy = np.average(np.equal(predictions, labels).astype(float))
    macro_f1 = f1_score(labels, predictions, average='macro')
    micro_f1 = f1_score(labels, predictions, average='micro')
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')

    opt_test_log = {
        ClassificationMetric.ACCURACY.name: accuracy,
        ClassificationMetric.MACRO_F1_SCORE.name: macro_f1,
        ClassificationMetric.MICRO_F1_SCORE.name: micro_f1,
        ClassificationMetric.PRECISION.name: precision,
        ClassificationMetric.RECALL.name: recall,
        'APPROX_POWER': np.average(power)
    }
    save_by_file_suffix([opt_test_log], opt_log_file)

    # Get the index of the best 'fixed' policy
    baseline_level_acc = np.average(np.isclose(baseline_predictions, baseline_labels).astype(float), axis=0)  # [L]
    budget_index = get_budget_index(power_estimates, budget, baseline_level_acc)
    acc_index = get_accuracy_index(accuracy, baseline_level_acc)

    budget_policy_acc, budget_policy_power = run_fixed_policy(labels=baseline_labels,
                                                              level_predictions=baseline_predictions,
                                                              policy_index=budget_index,
                                                              power_estimates=power_estimates,
                                                              noise=noise)

    accuracy_policy_acc, accuracy_policy_power = run_fixed_policy(labels=baseline_labels,
                                                                  level_predictions=baseline_predictions,
                                                                  policy_index=acc_index,
                                                                  power_estimates=power_estimates,
                                                                  noise=noise)

    cumulative_budget_policy_acc = np.cumsum(budget_policy_acc) / (times + 1)
    cumulative_accuracy_policy_acc = np.cumsum(accuracy_policy_acc) / (times + 1)
    budget_avg_power = np.cumsum(budget_policy_power) / (times + 1)
    accuracy_avg_power = np.cumsum(accuracy_policy_power) / (times + 1)
    
    adaptive_power = np.cumsum(power) / (times + 1)
    
    budget_policy_energy = np.cumsum(budget_policy_power)
    accuracy_policy_energy = np.cumsum(accuracy_policy_power)
    adaptive_energy = np.cumsum(power)

    # Plot the results
    with plt.style.context('ggplot'):
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(figsize=(16, 12), nrows=5, ncols=1, sharex=True)

        ax1.plot(times, desired_levels, label='true')
        ax1.legend()
        ax1.set_title('True Model Levels over Time')

        ax2.plot(times, errors, label='error')
        ax2.legend()
        ax2.set_title('Model Controller Error')

        ax3.plot(times, labels, label='true labels')
        ax3.legend()
        ax3.set_title('True Labels over Time')
        ax3.set_ylabel('Label Number')

        # Set ranges for formatting purposes
        acc_min = np.percentile(cumulative_budget_policy_acc, 0.5) - 0.05
        ax4.plot(times, cumulative_accuracy, label='Adaptive')
        ax4.plot(times, cumulative_budget_policy_acc, label='Budget Policy')
        ax4.plot(times, cumulative_accuracy_policy_acc, label='Accuracy Policy')
        ax4.legend()
        ax4.set_ylim((acc_min, 1.0))
        ax4.set_title('Model Accuracy over Time')

        # Plot the energy
        power_budget = [budget for _ in times]
        ax5.plot(times, adaptive_power, label='Adaptive')
        ax5.plot(times, budget_avg_power, label='Budget Policy')
        ax5.plot(times, accuracy_avg_power, label='Accuracy Policy')
        ax5.plot(times, power_budget, label='Budget')
        ax5.legend()
        ax5.set_title('Cumulative Average Power')
        ax5.set_ylabel('Power (mW)')
        ax5.set_xlabel('Time')

        plt.tight_layout()

        if output_folder is not None:
            make_dir(output_folder)
            output_file = os.path.join(output_folder, 'results_{0}.pdf'.format(budget))
            plt.savefig(output_file)
        else:
            plt.show()   


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--adaptive-model-path', type=str, required=True)
    parser.add_argument('--baseline-model-path', type=str, required=True)
    parser.add_argument('--precision', type=int, required=True)
    parser.add_argument('--dataset-folder', type=str, required=True)
    parser.add_argument('--budgets', type=float, nargs='+')
    parser.add_argument('--output-folder', type=str)
    parser.add_argument('--noise', type=float, default=1.0)
    parser.add_argument('--margin', type=float, default=0.001)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--controller', type=str, choices=['random', 'logistic'], default='logistic')
    args = parser.parse_args()

    # Validate the budget and margin
    budgets = args.budgets
    assert all([b > 0 for b in budgets]), 'Must have a positive budgets'

    margin = args.margin
    assert margin >= 0 and margin <= 1, 'Must have a margin in the range [0, 1]'

    # Create the adaptive model
    adaptive_model, dataset = get_serialized_info(args.adaptive_model_path, dataset_folder=args.dataset_folder)
    output_range = (0, adaptive_model.num_outputs - 1)

    # Get results from the adaptive model
    labels, level_predictions, states, level_accuracy, dataset_inputs = get_model_results(model=adaptive_model, dataset=dataset, shuffle=args.shuffle, series=DataSeries.TEST)

    # Create baseline model and get results
    baseline_model, _ = get_serialized_info(args.baseline_model_path, dataset_folder=args.dataset_folder)

    # TODO: Get the baseline model results
    base_labels, base_predictions = get_baseline_model_results(model=baseline_model, dataset=dataset, shuffle=args.shuffle, series=DataSeries.TEST)

    # Truncate the power readings
    power_estimates = get_power_for_levels(power=POWER, num_levels=adaptive_model.num_outputs)

    # Run the simulation for each budget
    for budget in budgets:
        power, errors, num_correct, predictions, desired_levels = run_simulation(labels=labels,
                                                                                 level_predictions=level_predictions,
                                                                                 states=dataset_inputs,
                                                                                 budget=budget,
                                                                                 controller_type=args.controller,
                                                                                 num_levels=adaptive_model.num_outputs,
                                                                                 num_classes=adaptive_model.metadata['num_classes'],
                                                                                 power_estimates=power_estimates,
                                                                                 noise=args.noise,
                                                                                 precision=args.precision,
                                                                                 model_path=args.adaptive_model_path,
                                                                                 dataset_folder=args.dataset_folder)
        
        # Plot and save the results
        plot_and_save(power=power,
                      errors=errors,
                      num_correct=num_correct,
                      predictions=predictions,
                      labels=labels,
                      desired_levels=desired_levels,
                      model_path=args.adaptive_model_path,
                      output_folder=args.output_folder,
                      power_estimates=power_estimates,
                      budget=budget,
                      controller_type=args.controller,
                      num_levels=adaptive_model.num_outputs,
                      level_accuracy=level_accuracy,
                      level_predictions=level_predictions,
                      noise=args.noise,
                      baseline_labels=base_labels,
                      baseline_predictions=base_predictions)
