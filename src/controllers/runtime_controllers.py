import numpy as np
from scipy import integrate
from typing import Tuple, Union, List, Dict

from controllers.controller_utils import clip
from controllers.power_utils import get_avg_power
from utils.constants import SMALL_NUMBER


SMOOTHING_FACTOR = 100
POWER_PRIOR_COUNT = 100


class PIDController:

    def __init__(self, kp: float, ki: float, kd: float, integral_bounds: Tuple[float, float]):
        self._kp = kp
        self._ki = ki
        self._kd = kd

        self._errors: List[float] = []
        self._times: List[float] = []
        self._integral_bounds = integral_bounds

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

        # Approximate the derivative term
        derivative = 0
        if len(self._errors) > 0:
            derivative = (error - self._errors[-1]) / (time - self._times[-1])

        self._errors.append(error)
        self._times.append(time)

        # Approximate the integral term using a trapezoid rule approximation
        integral = 0
        if len(self._errors) > 1:
            integral = integrate.trapz(self._errors, self._times)
            integral = clip(integral, bounds=self._integral_bounds)

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

    def __init__(self, kp: float, ki: float, kd: float, output_range: Tuple[int, int], integral_bounds: Tuple[float, float], budget: float, window: int):
        super().__init__(kp, ki, kd, integral_bounds)
        self._output_range = output_range
        self._budget = budget
        self._window = window
        self._rand = np.random.RandomState(seed=92)

    def plant_function(self, y_true: Tuple[float, float], y_pred: Tuple[float, float], control_error: float) -> float:
        # If within bounds, we don't apply and adjustments
        if y_pred >= y_true[0] and y_pred < y_true[1]:
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
                 panic_frac: float):
        self._prior_counts = prior_counts  # key: class index, value: array [L] counts for each level
        self._max_time = max_time
        self._budget = budget
        self._num_levels = num_levels
        self._num_classes = num_classes
        self._panic_time = int(panic_frac * max_time)
        self._level_counts = np.zeros(shape=(num_levels, ))
        self._observed_power = np.zeros(shape=(num_levels, ))
        self._seq_length = seq_length
        self._power_multiplier = int(seq_length / num_levels)

        # Estimate the power prior based on profiling
        self._prior_power = [get_avg_power(num_samples=level + 1, seq_length=seq_length, multiplier=self._power_multiplier) for level in range(num_levels)]
        self._prior_power = np.array(self._prior_power)

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
            return self._budget, self._budget

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
        expected_power = max(expected_power, power_estimates[0])  # We clip the power to the lowest level

        estimator_variance = 2 * (1.0 / time) * variance_rest
        estimator_std = np.sqrt(estimator_variance)

        # Upper and lower bounds as determined by one std from the mean
        return expected_power - estimator_std, expected_power + estimator_std

    def update(self, label: int, levels: int, power: float):
        self._observed_label_counts[label] += 1
        self._prior_counts[label][levels] += 1
        self._level_counts[levels] += 1
        self._observed_power[levels] += power
