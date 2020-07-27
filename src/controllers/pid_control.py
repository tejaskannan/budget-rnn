import numpy as np
import matplotlib.pyplot as plt
import math
import os.path
from argparse import ArgumentParser
from collections import defaultdict, namedtuple
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy import integrate
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
from utils.constants import OUTPUT, SMALL_NUMBER, INPUTS, OPTIMIZED_TEST_LOG_PATH, METADATA_PATH, HYPERS_PATH, PREDICTION, SEQ_LENGTH, DROPOUT_KEEP_RATE
from utils.adaptive_inference import normalize_logits, threshold_predictions
from utils.testing_utils import ClassificationMetric
from controllers.logistic_regression_controller import Controller, CONTROLLER_PATH, get_power_for_levels, POWER, RandomController, FixedController, BudgetWrapper


PowerEstimate = namedtuple('PowerEstimate', ['avg_power', 'fraction'])
SimulationResult = namedtuple('SimulationResult', ['adaptive_accuracy', 'adaptive_power', 'greedy_accuracy', 'greedy_power', 'randomized_accuracy', 'randomized_power', \
                                                   'adaptive_desired_levels', 'adaptive_controller_error'])
SMOOTHING_FACTOR = 100
POWER_PRIOR_COUNT = 100


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


def get_baseline_model_results(model: Model, dataset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    predictions: List[np.ndarray] = []
    batch_size = model.hypers.batch_size

    for batch_idx in range(0, dataset.shape[0], batch_size):
        start_idx, end_idx = batch_idx, batch_idx + batch_size
        data_batch = dataset[start_idx:end_idx]  # [B, T, D]

        feed_dict = {
            model.placeholders[INPUTS]: data_batch,
            model.placeholders[DROPOUT_KEEP_RATE]: 1.0
        }

        batch_predictions = model.execute(ops=[PREDICTION], feed_dict=feed_dict)

        predictions.append(batch_predictions[PREDICTION])

    predictions = np.vstack(predictions)

    return predictions


def clip(x: int, bounds: Tuple[int, int]) -> int:
    if x > bounds[1]:
        return bounds[1]
    elif x < bounds[0]:
        return bounds[0]
    return x


class PIDController:

    def __init__(self, kp: float, ki: float, kd: float):
        self._kp = kp
        self._ki = ki
        self._kd = kd

        self._errors: List[float] = []
        self._times: List[float] = []

    def errors(self) -> List[float]:
        return self._errors

    def times(self) -> List[float]:
        return self._times

    def plant_function(self, y_pred: Union[float, int], proportional_error: float, integral_error: float) -> float:
        raise NotImplementedError()

    def update(self, **kwargs):
        """
        A general update function to update any controller-specific parameters
        """
        pass

    def step(self, y_true: Tuple[float, float], y_pred: float, time: float) -> Union[float, int]:
        """
        Updates the controller and outputs the next control signal.
        """
        # Only add to error if it is out of the target bounds
        error = 0
        if y_pred < y_true[0]:
            error = y_true[0] - y_pred
        elif y_pred > y_true[1]:
            error = y_true[1] - y_pred

        self._errors.append(error)
        self._times.append(time)

        integral = 0
        if len(self._errors) > 1:
            integral = integrate.trapz(self._errors, self._times)
            
        derivative = (self._errors[-1] - error)

        derivative_error = self._kd * derivative
        integral_error = self._ki * integral
        proportional_error = self._kp * error
        control_error = proportional_error + integral_error + derivative_error

        self._errors.append(error)
        self._times.append(time)

        return self.plant_function(y_true, y_pred, control_error)

    def reset(self):
        """
        Resets the PI Controller.
        """
        self._errors = []
        self._times = []
        self._integral = 0.0


class BudgetController(PIDController):

    def __init__(self, kp: float, ki: float, kd: float, output_range: Tuple[int, int], budget: float, power_factor: float, window: int):
        super().__init__(kp, ki, kd)
        self._output_range = output_range
        self._budget = budget
        self._power_factor = power_factor
        self._window = window
        self._rand = np.random.RandomState(seed=92)

    def plant_function(self, y_true: Tuple[float, float], y_pred: Tuple[float, float], control_error: float) -> float:
        # If within bounds, we don't apply and adjustments
        if y_pred >= y_true[0] and y_pred < y_true[1]:
            return 0

        # Otherwise, we apply and offset proportional to the error
        step = control_error * self._power_factor
        return step


class BudgetDistribution:

    def __init__(self,
                 prior_counts: Dict[int, np.ndarray],
                 budget: float,
                 max_time: int,
                 num_levels: int,
                 num_classes: int,
                 panic_frac: float,
                 power: np.ndarray):
        self._prior_counts = prior_counts  # key: class index, value: array [L] counts for each level
        self._max_time = max_time
        self._budget = budget
        self._num_levels = num_levels
        self._num_classes = num_classes
        self._panic_time = int(panic_frac * max_time)
        self._prior_power = np.copy(power)  # [L] array of power readings
        self._level_counts = np.zeros_like(self._prior_power)
        self._observed_power = np.zeros_like(self._prior_power)

        # Estimated count of each label over the time window
        self._estimated_label_counts = np.zeros(shape=(num_classes, ))
        total_count = sum(np.sum(counts) for counts in prior_counts.values())
        for label, counts in prior_counts.items():
            normalized_count = (np.sum(counts) + SMOOTHING_FACTOR) / (total_count + self._num_classes * SMOOTHING_FACTOR)
            self._estimated_label_counts[label] += normalized_count * max_time

        self._observed_label_counts = np.zeros_like(self._estimated_label_counts)

    def get_budget(self, time: int) -> Tuple[float, float]:
        # Force the budget after the panic time is reached
        if time > self._panic_time:
            return budget, budget

        expected_rest = 0
        variance_rest = 0
        time_delta = self._max_time - time

        class_count_diff = np.maximum(self._estimated_label_counts - self._observed_label_counts, 0)
        estimated_remaining = np.sum(class_count_diff)

        # MLE estimate of the mean power, [L] array
        power_estimates = (POWER_PRIOR_COUNT * self._prior_power + self._observed_power) / (POWER_PRIOR_COUNT + self._level_counts)

        # We compute the MLE estimates for the mean and variance power given the observed samples
        # and the training set
        count_rest = 0
        for class_idx in range(self._num_classes):
            class_level_counts = self._prior_counts[class_idx]
            n_class = max(np.sum(class_level_counts), SMALL_NUMBER)

            # MLE estimate of the mean power
            power_mean = np.sum((class_level_counts * power_estimates) / (n_class))

            squared_diff = np.square(power_estimates - power_mean)
            power_var = np.sum((class_level_counts * squared_diff) / n_class)

            count_diff = max(self._estimated_label_counts[class_idx] - self._observed_label_counts[class_idx], 0)
            remaining_fraction = count_diff / estimated_remaining

            expected_rest += power_mean * remaining_fraction
            variance_rest += np.square(count_diff / time) * power_var

        expected_power = (1.0 / time) * (self._max_time * self._budget - time_delta * expected_rest)
        expected_power = max(expected_power, power_estimates[0])  # We clip the power to the lowest level (this is the lowest feasible amount)

        estimator_variance = 2 * (1.0 / time) * variance_rest
        estimator_std = np.sqrt(estimator_variance)

        # Upper and lower bounds as determined by one std from the mean
        return expected_power - estimator_std, expected_power + estimator_std

    def update(self, label: int, levels: int, power: float):
        self._observed_label_counts[label] += 1
        self._prior_counts[label][levels] += 1
        self._level_counts[levels] += 1
        self._observed_power[levels] += power


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
                     noise: Tuple[float, float]) -> Tuple[List[float], List[float]]:
    rand = np.random.RandomState(seed=42)

    correct: List[float] = []
    power: List[float] = []

    power_readings = interpolate_power(power_estimates, level_predictions.shape[1])

    max_time = labels.shape[0]
    for t in range(max_time):
        label = int(labels[t])
        pred = int(level_predictions[t, policy_index])
        correct.append(float(label == pred))

        p = power_readings[policy_index] + rand.normal(loc=noise[0], scale=noise[1])
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
                   dataset_inputs: np.ndarray,
                   adaptive_predictions: np.ndarray,
                   baseline_predictions: np.ndarray,
                   power_estimates: np.ndarray,
                   budget: int,
                   num_levels: int,
                   num_classes: int,
                   controller_window: float,
                   noise: Tuple[float, float],
                   precision: int,
                   model_path: str,
                   dataset_folder: str) -> SimulationResult:
    # Extract the time horizon
    max_time = level_predictions.shape[0]

    # Create the three different controllers: Adaptive, Randomized and Greedy
    save_folder, model_file_name = os.path.split(model_path)
    model_name = extract_model_name(model_file_name)
    adaptive_controller = Controller.load(os.path.join(save_folder, CONTROLLER_PATH.format(model_name)))
    adaptive_budget_controller = BudgetWrapper(controller=adaptive_controller,
                                               model_predictions=adaptive_predictions,
                                               power_estimates=power_estimates,
                                               max_time=max_time,
                                               num_classes=num_classes,
                                               budget=budget,
                                               seed=70)

    randomized_controller = RandomController(model_path=model_path,
                                             dataset_folder=dataset_folder,
                                             budgets=[budget],
                                             power=power_estimates)
    randomized_controller.fit(series=DataSeries.VALID)
    randomized_budget_controller = BudgetWrapper(controller=randomized_controller,
                                                 model_predictions=adaptive_predictions,
                                                 max_time=max_time,
                                                 power_estimates=power_estimates,
                                                 num_classes=num_classes,
                                                 budget=budget,
                                                 seed=71)

    # Create the greedy controller
    baseline_level_accuracy = np.average(np.isclose(baseline_predictions, np.expand_dims(labels, axis=1)), axis=0)
    baseline_index = np.argmax(baseline_level_accuracy)

    max_power = get_power_for_levels(power_estimates, num_levels)[-1]  # Power of the top-level model
    greedy_controller = FixedController(model_index=baseline_index)
    greedy_budget_controller = BudgetWrapper(controller=greedy_controller,
                                             model_predictions=baseline_predictions,
                                             max_time=max_time,
                                             power_estimates=interpolate_power(power_estimates, baseline_predictions.shape[1]),
                                             num_classes=num_classes,
                                             budget=budget,
                                             seed=72)

    # Execute model on the validation set and collect levels
    adaptive_level_accuracy = np.average((adaptive_predictions == np.expand_dims(labels, axis=1)).astype(float), axis=0)

    # Lists to save results
    adaptive_correct: List[float] = []
    randomized_correct: List[float] = []
    greedy_correct: List[float] = []

    adaptive_energy: List[float] = []
    randomized_energy: List[float] = []
    greedy_energy: List[float] = []

    adaptive_desired_levels: List[float] = []
    adaptive_controller_error: List[float] = []

    # Set random state for reproducible results
    rand = np.random.RandomState(seed=42)

    # Create the budget controller
    output_range = (0, num_levels - 1)
    budget_controller = BudgetController(kp=1.0,
                                         ki=(1.0 / 64.0),
                                         kd=(1.0 / 32.0),
                                         output_range=output_range,
                                         budget=budget,
                                         power_factor=1.0,
                                         window=controller_window)
    current_budget = (budget, budget)

    level_counts = np.zeros(shape=(num_levels, ))
    power_noise = rand.normal(loc=noise[0], scale=noise[1], size=(max_time, ))
    level_idx = np.arange(num_levels)

    # Create the power distribution
    prior_counts = estimate_label_counts(adaptive_controller, budget, num_levels, num_classes)
    budget_distribution = BudgetDistribution(prior_counts=prior_counts,
                                             budget=budget,
                                             max_time=max_time,
                                             num_levels=num_levels,
                                             num_classes=num_classes,
                                             panic_frac=0.95,
                                             power=power_estimates)

    budget_step = 0
    for t in range(max_time):
        # Perform inference using the adaptive controller
        # We first determine the number of states to collect
        adaptive_pred, adaptive_level = adaptive_budget_controller.predict_sample(inputs=dataset_inputs[t],
                                                                                  budget=budget + budget_step,
                                                                                  noise=power_noise[t],
                                                                                  current_time=t)

        # Save adaptive controller results
        adaptive_correct.append(float(adaptive_pred == labels[t]))
        adaptive_energy.append(adaptive_budget_controller.get_consumed_energy())
        adaptive_controller_error.append(budget + budget_step)
        adaptive_desired_levels.append(adaptive_level)

        # Update the budget distribution
        p = power_estimates[adaptive_level] + power_noise[t]
        budget_distribution.update(label=adaptive_pred, levels=adaptive_level, power=p)

        # Perform inference with the randomized policy
        randomized_pred, randomized_level = randomized_budget_controller.predict_sample(inputs=dataset_inputs[t],
                                                                                        budget=budget,
                                                                                        noise=power_noise[t],
                                                                                        current_time=t)
        randomized_energy.append(randomized_budget_controller.get_consumed_energy())
        randomized_correct.append(float(randomized_pred == labels[t]))

        # Perform inference with the greedy baseline policy
        greedy_pred, greedy_level = greedy_budget_controller.predict_sample(inputs=dataset_inputs[t],
                                                                            budget=budget,
                                                                            noise=power_noise[t],
                                                                            current_time=t)
        greedy_energy.append(greedy_budget_controller.get_consumed_energy())
        greedy_correct.append(float(greedy_pred == labels[t]))

        # Get new control signals
        if (t + 1) % controller_window == 0:
            current_budget = budget_distribution.get_budget(t+1)
            adaptive_power = adaptive_energy[-1] / t
            budget_step = budget_controller.step(y_true=current_budget, y_pred=adaptive_power, time=t)

    # Create the simulation result tuple
    times = np.arange(max_time) + 1

    result = SimulationResult(adaptive_accuracy=np.cumsum(adaptive_correct) / times,
                              adaptive_power=adaptive_energy / times,
                              greedy_accuracy=np.cumsum(greedy_correct) / times,
                              greedy_power=greedy_energy / times,
                              randomized_accuracy=np.cumsum(randomized_correct) / times,
                              randomized_power=randomized_energy / times,
                              adaptive_desired_levels=adaptive_desired_levels,
                              adaptive_controller_error=adaptive_controller_error)
    return result


def plot_and_save(sim_result: SimulationResult,
                  labels: List[float],
                  adaptive_model_path: str,
                  baseline_model_path: str,
                  output_folder: Optional[str],
                  budget: int,
                  num_levels: int,
                  noise: Tuple[float, float],
                  baseline_predictions: np.ndarray,
                  power_estimates: np.ndarray,
                  should_plot: bool):
    # Create optimized test log path for the adaptive policy
    save_folder, model_file_name = os.path.split(adaptive_model_path)
    model_name = extract_model_name(model_file_name)
    adaptive_log_file = os.path.join(save_folder, OPTIMIZED_TEST_LOG_PATH.format('adaptive', 'power', budget, model_name))

    adaptive_test_log = {
        ClassificationMetric.ACCURACY.name: sim_result.adaptive_accuracy[-1],
        'APPROX_POWER': sim_result.adaptive_power[-1]
    }
    save_by_file_suffix([adaptive_test_log], adaptive_log_file)

    # Create optimized test log path for the randomized policy
    randomized_log_file = os.path.join(save_folder, OPTIMIZED_TEST_LOG_PATH.format('randomized', 'power', budget, model_name))

    randomized_test_log = {
        ClassificationMetric.ACCURACY.name: sim_result.randomized_accuracy[-1],
        'APPROX_POWER': sim_result.randomized_power[-1]
    }
    save_by_file_suffix([randomized_test_log], randomized_log_file)

    # Create optimized test log path for the greedy
    save_folder, model_file_name = os.path.split(baseline_model_path)
    model_name = extract_model_name(model_file_name)
    greedy_log_file = os.path.join(save_folder, OPTIMIZED_TEST_LOG_PATH.format('greedy', 'power', budget, model_name))

    print('Greedy Accuracy: {0:.5f}, Greedy Power: {1:.5f}'.format(sim_result.greedy_accuracy[-1], sim_result.greedy_power[-1]))
    print('Randomized Accuracy: {0:.5f}, Randomized Power: {1:.5f}'.format(sim_result.randomized_accuracy[-1], sim_result.randomized_power[-1]))
    print('Adaptive Accuracy: {0:.5f}, Adaptive Power: {1:.5f}'.format(sim_result.adaptive_accuracy[-1], sim_result.adaptive_power[-1]))

    greedy_test_log = {
        ClassificationMetric.ACCURACY.name: sim_result.greedy_accuracy[-1],
        'APPROX_POWER': sim_result.greedy_power[-1]
    }
    save_by_file_suffix([greedy_test_log], greedy_log_file)

    # This is for convenience for testing
    if not should_plot:
        return

    # Get the index of the best 'fixed' policy
    baseline_level_acc = np.average(np.isclose(baseline_predictions, np.expand_dims(labels, axis=1)).astype(float), axis=0)  # [L]
    budget_index = get_budget_index(power_estimates, budget, baseline_level_acc)
    acc_index = get_accuracy_index(sim_result.adaptive_accuracy[-1], baseline_level_acc)

    budget_policy_acc, budget_policy_power = run_fixed_policy(labels=labels,
                                                              level_predictions=baseline_predictions,
                                                              policy_index=budget_index,
                                                              power_estimates=power_estimates,
                                                              noise=noise)

    accuracy_policy_acc, accuracy_policy_power = run_fixed_policy(labels=labels,
                                                                  level_predictions=baseline_predictions,
                                                                  policy_index=acc_index,
                                                                  power_estimates=power_estimates,
                                                                  noise=noise)

    times = np.arange(sim_result.adaptive_accuracy.shape[0]) + 1
    cumulative_budget_policy_acc = np.cumsum(budget_policy_acc) / (times + 1)
    cumulative_accuracy_policy_acc = np.cumsum(accuracy_policy_acc) / (times + 1)
    budget_avg_power = np.cumsum(budget_policy_power) / (times + 1)
    accuracy_avg_power = np.cumsum(accuracy_policy_power) / (times + 1)

    # Plot the results
    with plt.style.context('ggplot'):
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(figsize=(16, 12), nrows=5, ncols=1, sharex=True)

        ax1.plot(times, sim_result.adaptive_desired_levels, label='true')
        ax1.legend()
        ax1.set_title('True Model Levels for Adaptive Policy')

        ax2.plot(times, sim_result.adaptive_controller_error, label='error')
        ax2.legend()
        ax2.set_title('Adaptive Controller Error')

        ax3.plot(times, labels, label='true labels')
        ax3.legend()
        ax3.set_title('True Labels over Time')
        ax3.set_ylabel('Label Number')

        # Set ranges for formatting purposes
        acc_min = np.percentile(cumulative_budget_policy_acc, 0.5) - 0.05
        ax4.plot(times, sim_result.adaptive_accuracy, label='Adaptive')
        ax4.plot(times, sim_result.randomized_accuracy, label='Randomized')
        ax4.plot(times, sim_result.greedy_accuracy, label='Greedy Baseline')
        ax4.plot(times, cumulative_budget_policy_acc, label='Budget Policy')
        ax4.plot(times, cumulative_accuracy_policy_acc, label='Accuracy Policy')
        ax4.legend()
        ax4.set_ylim((acc_min, 1.0))
        ax4.set_title('Model Accuracy over Time')

        # Plot the energy
        power_budget = [budget for _ in times]
        ax5.plot(times, sim_result.adaptive_power, label='Adaptive')
        ax5.plot(times, sim_result.randomized_power, label='Randomized')
        ax5.plot(times, sim_result.greedy_power, label='Greedy Baseline')
        ax5.plot(times, budget_avg_power, label='Budget Policy')
        ax5.plot(times, accuracy_avg_power, label='Accuracy Policy')
        ax5.plot(times, power_budget, label='Budget')
        ax5.legend()
        ax5.set_ylim((budget - 5, power_estimates[-1] + 5))

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
    parser.add_argument('--controller-window', type=int, default=20)
    parser.add_argument('--noise-loc', type=float, default=0.0)
    parser.add_argument('--noise-scale', type=float, default=1.0)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--skip-plotting', action='store_true')
    args = parser.parse_args()

    # Validate arguments
    budgets = args.budgets
    assert all([b > 0 for b in budgets]), 'Must have a positive budgets'

    assert args.controller_window > 0, 'Must have a positive controller window'
    assert args.noise_scale > 0, 'Must have a positive noise scale'

    # Create the adaptive model
    adaptive_model, dataset = get_serialized_info(args.adaptive_model_path, dataset_folder=args.dataset_folder)
    output_range = (0, adaptive_model.num_outputs - 1)

    # Get results from the adaptive model
    labels, level_predictions, states, level_accuracy, dataset_inputs = get_model_results(model=adaptive_model, dataset=dataset, shuffle=args.shuffle, series=DataSeries.TEST)

    # Create baseline model and get results
    baseline_model, _ = get_serialized_info(args.baseline_model_path, dataset_folder=args.dataset_folder)
    base_predictions = get_baseline_model_results(model=baseline_model, dataset=dataset_inputs)

    # TODO: Add a randomized baseline where we randomly drop samples to meet the budget (using the 
    # number of fixed policy samples, pick an (ordered) subset which matches this amount)

    # Truncate the power readings
    power_estimates = get_power_for_levels(power=POWER, num_levels=adaptive_model.num_outputs)

    # Run the simulation for each budget
    for budget in budgets:
        print('Starting Budget: {0}'.format(budget))

        result = run_simulation(labels=labels,
                                dataset_inputs=dataset_inputs,
                                adaptive_predictions=level_predictions,
                                budget=budget,
                                baseline_predictions=base_predictions,
                                num_levels=adaptive_model.num_outputs,
                                num_classes=adaptive_model.metadata['num_classes'],
                                power_estimates=power_estimates,
                                controller_window=args.controller_window,
                                noise=(args.noise_loc, args.noise_scale),
                                precision=args.precision,
                                model_path=args.adaptive_model_path,
                                dataset_folder=args.dataset_folder)

        # Plot and save the results
        plot_and_save(sim_result=result,
                      labels=labels,
                      adaptive_model_path=args.adaptive_model_path,
                      baseline_model_path=args.baseline_model_path,
                      output_folder=args.output_folder,
                      power_estimates=power_estimates,
                      budget=budget,
                      num_levels=adaptive_model.num_outputs,
                      noise=(args.noise_loc, args.noise_scale),
                      baseline_predictions=base_predictions,
                      should_plot=not args.skip_plotting)
