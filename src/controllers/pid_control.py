import numpy as np
import matplotlib.pyplot as plt
import math
import os.path
from argparse import ArgumentParser
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Tuple, List, Union, Optional

from models.adaptive_model import AdaptiveModel
from dataset.dataset import DataSeries, Dataset
from utils.rnn_utils import get_logits_name, get_states_name
from utils.file_utils import extract_model_name, read_by_file_suffix, save_by_file_suffix, make_dir
from utils.np_utils import min_max_normalize, round_to_precision
from utils.constants import OUTPUT, SMALL_NUMBER, INPUTS, OPTIMIZED_TEST_LOG_PATH
from utils.adaptive_inference import normalize_logits, threshold_predictions
from utils.testing_utils import ClassificationMetric
from threshold_optimization.optimize_thresholds import get_serialized_info
from logistic_regression_controller import Controller, CONTROLLER_PATH


POWER = np.array([24.085, 32.776, 37.897, 43.952, 48.833, 50.489, 54.710, 57.692, 59.212, 59.251])


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


class RandomController(PIController):
    
    def __init__(self, budget: float):
        super().__init__(0.0, 0.0)
        self._budget = budget

        power_array = np.array(POWER)
        weights = np.linalg.lstsq(power_array.reshape(1, -1), np.array([self._budget]))[0]

        self._weights = weights / np.sum(weights)
        self._indices = np.arange(start=0, stop=len(POWER))
        np.random.seed(42)  # For reproducible results
    
    def step(self, y_true: Union[float, int], y_pred: Union[float, int], time: float) -> Union[float, int]:
        return int(np.random.choice(self._indices, size=1, p=self._weights))


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

        if power >= lower_limit and power <= upper_limit:
            return 0  # By returning the highest # of levels, we allow the model controller to control
        
        sign = 1
        if power > upper_limit:
            sign = -1

        step = int(math.floor(sign * step))

        if step == 0:
            return sign
        return step

# Function to get model results
def get_model_results(model: AdaptiveModel, dataset: Dataset, shuffle: bool):
    labels: List[np.ndarray] = []
    level_predictions: List[np.ndarray] = []
    states: List[np.ndarray] = []

    logit_ops = [get_logits_name(i) for i in range(model.num_outputs)]
    state_ops = [get_states_name(i) for i in range(model.num_outputs)]
    data_generator = dataset.minibatch_generator(series=DataSeries.TEST,
                                                 batch_size=model.hypers.batch_size,
                                                 metadata=model.metadata,
                                                 should_shuffle=shuffle)
    for batch_num, batch in enumerate(data_generator):
        # Compute the predicted log probabilities
        feed_dict = model.batch_to_feed_dict(batch, is_train=False)
        model_results = model.execute(feed_dict, logit_ops + state_ops)

        first_states = np.concatenate([np.expand_dims(np.squeeze(model_results[op][0]), axis=1) for op in state_ops], axis=1)
        states.append(first_states)

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

    level_acc = np.equal(level_predictions, np.expand_dims(labels, axis=-1)).astype(float)
    level_acc = np.average(level_acc, axis=0)

    return labels, level_predictions, states, level_acc


def get_budget_index(budget: int, num_levels: int) -> int:
    fixed_index = 0
    while fixed_index < num_levels and POWER[fixed_index] < budget:
        fixed_index += 1

    return fixed_index - 1


def get_accuracy_index(system_acc: float, level_accuracy: np.ndarray) -> int:
    num_levels = len(level_accuracy)

    policy_idx = 0
    while policy_idx < num_levels and level_accuracy[policy_idx] < system_acc:
        policy_idx += 1

    return min(policy_idx, num_levels - 1)


def run_fixed_policy(labels: np.ndarray,
                     level_predictions: int,
                     policy_index: int,
                     noise: float) -> Tuple[List[float], List[float]]:
    rand = np.random.RandomState(seed=42)

    correct: List[float] = []
    power: List[float] = []

    max_time = labels.shape[0]
    for t in range(max_time):
        label = int(labels[t])
        pred = int(level_predictions[t, policy_index])
        correct.append(float(label == pred))

        p = POWER[policy_index] + rand.uniform(low=-noise, high=noise)
        power.append(p)

    return correct, power


def run_simulation(labels: np.ndarray,
                   level_predictions: np.ndarray,
                   states: np.ndarray,
                   budget: int,
                   controller_type: str,
                   num_levels: int,
                   noise: float,
                   precision: int,
                   model_path: str):
    # Create the model controller
    if controller_type == 'random':
        controller = RandomController(budget=budget)
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
    anneal_rate = np.exp((1.0 / max_time) * np.log(end_margin / start_margin))

    # Create the budget controller
    output_range = (0, num_levels - 1)
    budget_controller = BudgetController(kp=1.0, ki=0.0625, output_range=output_range, budget=budget, margin=start_margin, power_factor=1.0)

    window_size = 10
    distribution_margin = 1.0
    end_distribution_margin = 1e-3
    # distribution_anneal_rate = np.exp((1.0 / max_time) * np.log(end_distribution_margin / distribution_margin))
    distribution_anneal_rate = 1.0
    thresholds = np.copy(controller.get_thresholds(budget=budget))
    target_distribution = controller.get_avg_level_counts(budget=budget)

    print('Target Distribution: {0}'.format(target_distribution))

   # init_margin = 0.5
   # panic_time = 0.9 * max_time
   # current_budget = budget * (1 + init_margin)
   # anneal_rate = np.exp((1.0 / panic_time) * np.log(budget / current_budget))

    level_counts = np.zeros(shape=thresholds.shape)
    fp_one = 1 << precision
    power_noise = rand.uniform(low=-noise, high=noise, size=(max_time, ))
    level_idx = np.arange(num_levels)

    for t in range(max_time):
        # Use the control model to determine the number of states to collect
        y_pred_model = controller.predict_sample(states=states[t], budget=budget, thresholds=thresholds)

        # Make adjustments based on observed power
        avg_power = np.average(power) if len(power) > 0 else 0
        avg_power = np.clip(avg_power, a_min=budget, a_max=None)
        
        budget_step = budget_controller.step(y_true=budget, y_pred=avg_power, time=t)
        budget_controller.update(anneal_rate=anneal_rate)

        # Form predicted levels using both controllers
        y_pred = clip(y_pred_model + budget_step, bounds=output_range)

        # Compute the (noisy) power consumption for using this number of levels
        p = POWER[y_pred] + power_noise[t]

        power.append(p)
        errors.append(y_pred_model - y_pred)
        desired_levels.append(y_pred_model)

        model_prediction = level_predictions[t, y_pred]
        predictions.append(float(model_prediction))
        num_correct.append(float(model_prediction == labels[t]))

        levels_per_label[model_prediction].append(y_pred)
        level_counts[y_pred_model] += 1

        # Adjust the thresholds
        #empirical_level_distribution = level_counts / (t + 1)

        #adjustments = np.zeros_like(thresholds)
        #for level in range(num_levels - 1):
        #    # Positive if we have too many, negative if we have too few
        #    level_diff = empirical_level_distribution[level] - target_distribution[level]
        #    proportional_error = int(fp_one * level_diff) / fp_one

        #    # This level has too few samples. We decrease its threshold
        #    # and increase the thresholds of all previous levels.
        #    if level_diff < -distribution_margin:
        #        if level > 0:
        #            adjustments[level-1] += (1.0 / fp_one)
        #        adjustments[level] += proportional_error
        #    elif level_diff > distribution_margin:
        #        if level > 0:
        #            adjustments[level-1] -= 1.0 / fp_one
        #        adjustments[level] += proportional_error

        #distribution_margin = distribution_margin * distribution_anneal_rate
        #thresholds = np.clip(thresholds + adjustments, a_min=0, a_max=1)

    print('Final Distribution: {0}'.format(level_counts / max_time))
    print('Starting Thresholds: {0}'.format(controller.get_thresholds(budget=budget)))
    print('Ending Thresholds: {0}'.format(thresholds))
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
                  noise: float):
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
    budget_index = get_budget_index(budget, num_levels)
    acc_index = get_accuracy_index(accuracy, level_accuracy)

    budget_policy_acc, budget_policy_power = run_fixed_policy(labels=labels,
                                                              level_predictions=level_predictions,
                                                              policy_index=budget_index,
                                                              noise=noise)

    accuracy_policy_acc, accuracy_policy_power = run_fixed_policy(labels=labels,
                                                                  level_predictions=level_predictions,
                                                                  policy_index=acc_index,
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

        ax4.plot(times, cumulative_accuracy, label='Adaptive')
        ax4.plot(times, cumulative_budget_policy_acc, label='Budget Policy')
        ax4.plot(times, cumulative_accuracy_policy_acc, label='Accuracy Policy')
        ax4.legend()
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
    parser.add_argument('--model-path', type=str, required=True)
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
    model, dataset, test_log = get_serialized_info(args.model_path, dataset_folder=args.dataset_folder)
    output_range = (0, model.num_outputs - 1)

    # Get results from the model
    labels, level_predictions, states, level_accuracy = get_model_results(model=model, dataset=dataset, shuffle=args.shuffle)

    # Run the simulation for each budget
    for budget in budgets:
        power, errors, num_correct, predictions, desired_levels = run_simulation(labels=labels,
                                                                                 level_predictions=level_predictions,
                                                                                 states=states,
                                                                                 budget=budget,
                                                                                 controller_type=args.controller,
                                                                                 num_levels=model.num_outputs,
                                                                                 noise=args.noise,
                                                                                 precision=args.precision,
                                                                                 model_path=args.model_path)
        # Plot and save the results
        plot_and_save(power=power,
                      errors=errors,
                      num_correct=num_correct,
                      predictions=predictions,
                      labels=labels,
                      desired_levels=desired_levels,
                      model_path=args.model_path,
                      output_folder=args.output_folder,
                      budget=budget,
                      controller_type=args.controller,
                      num_levels=model.num_outputs,
                      level_accuracy=level_accuracy,
                      level_predictions=level_predictions,
                      noise=args.noise)
