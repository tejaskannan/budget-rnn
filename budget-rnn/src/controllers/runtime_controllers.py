import numpy as np
from scipy import integrate
from typing import Tuple, Union, List, Dict
from collections import deque, defaultdict, namedtuple

from controllers.controller_utils import clip
from controllers.power_utils import PowerType, PowerSystem
from utils.constants import SMALL_NUMBER


ConfidenceBounds = namedtuple('ConfidenceBounds', ['lower', 'upper'])
SMOOTHING_FACTOR = 1
POWER_PRIOR_COUNT = 1
POWER_WINDOW = 100


class PIDController:

    def __init__(self, kp: float, ki: float, kd: float, integral_bounds: Tuple[float, float], integral_window: int):
        self._kp = kp
        self._ki = ki
        self._kd = kd

        self._errors: deque = deque()
        self._times: deque = deque()
        self._integral_bounds = integral_bounds
        self._integral_window = integral_window

    def plant_function(self, y_true: Tuple[float, float], y_pred: float, control_error: float) -> float:
        raise NotImplementedError()

    def step(self, y_true: Tuple[float, float], y_pred: float, time: float) -> Union[float, int]:
        """
        Updates the controller and outputs the next control signal.
        """
        # Only add to error if it is out of the target bounds
        error = 0.0
        if y_pred < y_true[0] - SMALL_NUMBER:
            error = y_true[0] - y_pred
        elif y_pred > y_true[1] + SMALL_NUMBER:
            error = y_true[1] - y_pred

        # Append error and clip to window
        self._errors.append(error)
        self._times.append(time)

        if len(self._errors) > self._integral_window:
            self._errors.popleft()

        if len(self._times) > self._integral_window:
            self._times.popleft()

        # Approximate the integral term using a trapezoid rule approximation
        integral = integrate.trapz(self._errors, dx=1)
        integral = clip(integral, bounds=self._integral_bounds)

        # Approximate the derivative term using the average over the window
        derivative = 0.0
        if len(self._errors) > 1:
            derivative_sum = 0
            for i in range(1, len(self._errors)):
                de = self._errors[i] - self._errors[i-1]
                dt = self._times[i] - self._times[i-1]
                derivative_sum = de / dt

            derivative = derivative_sum / (len(self._errors) - 1)

        derivative_error = self._kd * derivative
        integral_error = self._ki * integral
        proportional_error = self._kp * error
        control_error = proportional_error + integral_error + derivative_error

        return self.plant_function(y_true, y_pred, control_error)

    def reset(self):
        """
        Resets the PI Controller.
        """
        self._errors = deque()
        self._times = deque()


class BudgetController(PIDController):

    def plant_function(self, y_true: Tuple[float, float], y_pred: float, control_error: float) -> float:
        # If within bounds, we don't apply and adjustments
        if y_pred >= y_true[0] and y_pred <= y_true[1]:
            return 0

        # Otherwise, we apply an offset proportional to the error
        return control_error


class BudgetDistribution:

    def __init__(self,
                 prior_counts: Dict[int, np.ndarray],
                 budget: float,
                 max_time: int,
                 num_levels: int,
                 seq_length: int,
                 num_classes: int,
                 power_system: PowerSystem):
        self._prior_counts = prior_counts  # key: class index, value: array [L] counts for each level
        self._max_time = max_time
        self._num_levels = num_levels
        self._num_classes = num_classes

        self._budget = budget

        # Initialize variables for budget distribution tracking
        self._observed_power: deque = deque()
        self._seq_length = seq_length

        # Estimate the power prior based on profiling
        self._prior_power = power_system.get_power_estimates()  # [L]

        # Estimated count of each label over the time window
        self._estimated_label_counts = np.zeros(shape=(num_classes, ))
        total_count = sum(np.sum(counts) for counts in prior_counts.values())
        for label, counts in prior_counts.items():
            normalized_count = (np.sum(counts) + SMOOTHING_FACTOR) / (total_count + self._num_classes * SMOOTHING_FACTOR)
            self._estimated_label_counts[label] += normalized_count * max_time

        self._observed_label_counts = np.zeros_like(self._estimated_label_counts)

    def get_budget(self, time: int) -> ConfidenceBounds:
        expected_rest = 0
        variance_rest = 0
        time_delta = self._max_time - time

        class_count_diff = np.maximum(self._estimated_label_counts - self._observed_label_counts, 0)
        estimated_remaining = np.sum(class_count_diff)

        # MLE estimate of the mean power, [L] array
        level_counts = np.zeros((self._num_levels, ))
        observed_power = np.zeros((self._num_levels, ))
        for level, power in self._observed_power:
            level_counts[level] += 1
            observed_power[level] += power

        power_estimates = (POWER_PRIOR_COUNT * self._prior_power + observed_power) / (POWER_PRIOR_COUNT + level_counts)

        # We compute the MLE estimates for the mean and variance power given the observed samples
        # and the training set
        count_rest = 0
        for class_idx in range(self._num_classes):
            class_level_counts = self._prior_counts[class_idx]
            n_class = max(np.sum(class_level_counts), SMALL_NUMBER)

            # MLE estimate of the mean power
            power_mean = np.sum((class_level_counts * power_estimates) / n_class)

            # MLE estimate of the power variance
            squared_diff = np.square(power_estimates - power_mean)
            power_var = np.sum((class_level_counts * squared_diff) / n_class)

            # Estimate the fraction of remaining samples which should belong to this class
            remaining_fraction = class_count_diff[class_idx] / estimated_remaining

            expected_rest += power_mean * remaining_fraction
            variance_rest += np.square(class_count_diff[class_idx] / time) * power_var

        target = self._budget
        expected_power = (1.0 / time) * (self._max_time * target - time_delta * expected_rest)

        expected_power = clip(expected_power, (power_estimates[0], power_estimates[-1]))  # Clip the power to the feasible range

        estimator_variance = (1.0 / time) * variance_rest
        estimator_std = np.sqrt(estimator_variance)

        # Upper and lower bounds as determined by one std from the mean
        return ConfidenceBounds(lower=expected_power - estimator_std, upper=expected_power + estimator_std)

    def update(self, label: int, level: int, power: float):
        self._observed_label_counts[label] += 1
        self._prior_counts[label][level] += 1

        self._observed_power.append((level, power))
        if len(self._observed_power) > POWER_WINDOW:
            self._observed_power.popleft()
