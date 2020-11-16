import numpy as np
import os.path
import time
from collections import namedtuple
from enum import Enum, auto
from typing import List, Dict, Any, Tuple

from controllers.controller_utils import clip, get_budget_index, ModelResults
# from controllers.power_utils import get_power_estimates, get_avg_power_multiple
from controllers.power_utils import PowerType, make_power_system
from controllers.model_controllers import AdaptiveController, FixedController, RandomController, BudgetWrapper
from controllers.model_controllers import CONTROLLER_PATH, levels_to_execute, classification_for_levels, MultiModelController
from controllers.runtime_controllers import PIDController, BudgetController, BudgetDistribution
from dataset.dataset import DataSeries, Dataset
from models.adaptive_model import AdaptiveModel
from models.tf_model import TFModel
from utils.file_utils import extract_model_name
from utils.loading_utils import get_hyperparameters
from utils.constants import INPUTS, OUTPUT, SEQ_LENGTH, NUM_CLASSES, SEQ_LENGTH, SMALL_NUMBER


# Constants
KP = 1.0
KD = 1.0 / 32.0
KI = 1.0 / 8.0
WINDOW_SIZE = 20
INTEGRAL_BOUNDS = (-16, 16)
INTEGRAL_WINDOW = 4 * WINDOW_SIZE
POWER_WINDOW = 1000


# Defines the type of runtime system
class SystemType(Enum):
    ADAPTIVE = auto()
    RANDOMIZED = auto()
    GREEDY = auto()
    FIXED_MAX_ACCURACY = auto()
    FIXED_UNDER_BUDGET = auto()


class RuntimeSystem:

    def __init__(self,
                 valid_results: ModelResults,
                 test_results: ModelResults,
                 system_type: SystemType,
                 model_path: str,
                 dataset_folder: str,
                 num_classes: int,
                 num_levels: int,
                 seq_length: int,
                 power_system_type: PowerType):
        self._system_type = system_type
        self._test_results = test_results
        self._valid_results = valid_results

        self._model_path = model_path
        self._dataset_folder = dataset_folder

        self._num_classes = num_classes
        self._num_levels = num_levels
        self._seq_length = seq_length
        self._power_system_type = power_system_type

        save_folder, model_file_name = os.path.split(model_path)
        model_name = extract_model_name(model_file_name)

        self._model_name = model_name
        self._save_folder = save_folder

        # Results from the testing set. These are precomputed for efficiency.
        self._level_predictions = test_results.predictions
        self._test_accuracy = test_results.accuracy  # [L]
        self._stop_probs = test_results.stop_probs
        self._labels = test_results.labels

        # Results from the validation set. These are used by a few controllers
        # to select the model or model level
        self._valid_results = valid_results
        self._valid_accuracy = valid_results.accuracy  # [L]
        self._valid_predictions = valid_results.predictions  # [N, L]
        self._valid_stop_probs = valid_results.stop_probs  # [N, L]
        self._valid_labels = valid_results.labels  # [N]

        self._budget_controller = None

        model_type = model_name.split('-')[0]
        if model_type.lower().startswith('sample'):
            hypers = get_hyperparameters(model_path)
            stride_length = hypers.model_params['stride_length']
            model_type = '{0}({1})'.format(model_type, stride_length)

        self._name = '{0} {1}'.format(model_type, system_type.name)

        # For some systems, we can load the controller now. Otherwise, we wait until later
        if self._system_type == SystemType.ADAPTIVE:
            controller_path = CONTROLLER_PATH.format(self._power_system_type.name.lower(), model_name)
            self._controller = AdaptiveController.load(os.path.join(save_folder, controller_path),
                                                       dataset_folder=dataset_folder,
                                                       model_path=model_path)

            # Load validation accuracy
            self._controller.load_validation_accuracy(validation_accuracy=valid_results.accuracy)
        else:
            self._controller = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def save_folder(self) -> str:
        return self._save_folder

    @property
    def system_type(self) -> SystemType:
        return self._system_type

    def get_power(self) -> np.ndarray:
        assert self._budget_controller is not None, 'Must have a budget controller'
        return np.array(self._budget_controller.power)

    def get_energy(self) -> np.ndarray:
        assert self._budget_controller is not None
        return np.cumsum(self._budget_controller.power)

    def get_num_correct(self) -> np.ndarray:
        return np.cumsum(self._num_correct)

    def get_target_budgets(self) -> np.ndarray:
        return np.array(self._target_budgets).reshape(-1)

    def get_levels(self) -> np.ndarray:
        return np.array(self._levels).reshape(-1)

    def init_for_budget(self, budget: float, max_time: int):
        # Make controller based on the model type
        if self._system_type == SystemType.RANDOMIZED:
            self._controller = RandomController(budgets=[budget],
                                                seq_length=self._seq_length,
                                                num_levels=self._num_levels,
                                                power_system_type=self._power_system_type)
            self._controller.fit(series=None)
        elif self._system_type == SystemType.GREEDY:
            level = np.argmax(self._valid_accuracy)
            self._controller = FixedController(model_index=level,
                                               num_levels=self._num_levels,
                                               seq_length=self._seq_length,
                                               power_system_type=self._power_system_type)
        elif self._system_type in (SystemType.FIXED_UNDER_BUDGET, SystemType.FIXED_MAX_ACCURACY):
            allow_violations = self._system_type == SystemType.FIXED_MAX_ACCURACY

            # Create the fixed policy based on the model type. Skip RNNs use a similar strategy as those seen in
            # other model types. For Skip RNNs, however, the policy applies to model selection as opposed to
            # sample size selection.
            model_name = self._model_name.lower()
            if ('skip_rnn' in model_name) or ('phased_rnn' in model_name):
                self._controller = MultiModelController(sample_counts=self._valid_stop_probs,
                                                        model_accuracy=self._valid_accuracy,
                                                        seq_length=self._seq_length,
                                                        max_time=max_time,
                                                        allow_violations=allow_violations,
                                                        power_system_type=self._power_system_type)
            else:
                power_system = make_power_system(mode=self._power_system_type,
                                                 num_levels=self._num_levels,
                                                 seq_length=self._seq_length)

                level = get_budget_index(budget=budget,
                                         valid_accuracy=self._valid_accuracy,
                                         max_time=max_time,
                                         power_estimates=power_system.get_power_estimates(),
                                         allow_violations=allow_violations)
                self._controller = FixedController(model_index=level,
                                                   num_levels=self._num_levels,
                                                   seq_length=self._seq_length,
                                                   power_system_type=self._power_system_type)
        elif self._system_type == SystemType.ADAPTIVE:
            # Make the budget distribution and PID controller
            self._pid_controller = BudgetController(kp=KP,
                                                    ki=KI,
                                                    kd=KD,
                                                    integral_bounds=INTEGRAL_BOUNDS,
                                                    integral_window=INTEGRAL_WINDOW)

            # Create the power distribution. TODO: Remove (explicit) dependence on the validation results.
            # Instead, we should mix the label counts from the bounded sides using a weighted average.
            prior_counts = self._controller.estimate_level_distribution(budget=budget)

            self._budget_distribution = BudgetDistribution(prior_counts=prior_counts,
                                                           budget=budget,
                                                           max_time=max_time,
                                                           num_levels=self._num_levels,
                                                           num_classes=self._num_classes,
                                                           seq_length=self._seq_length,
                                                           power_system_type=self._power_system_type)

        # Apply budget wrapper to the controller
        assert self._controller is not None, 'Must have a valid controller'
        self._budget_controller = BudgetWrapper(controller=self._controller,
                                                model_predictions=self._level_predictions,
                                                max_time=max_time,
                                                num_classes=self._num_classes,
                                                num_levels=self._num_levels,
                                                seq_length=self._seq_length,
                                                budget=budget)
        self._budget_step = 0
        self._current_budget: Tuple[float, float] = (budget, budget)
        self._num_correct: List[float] = []
        self._target_budgets: List[float] = []
        self._levels: List[int] = []

    def step(self, budget: float, power_noise: float, t: int):
        assert self._budget_controller is not None, 'Must call init_for_budget() first'
        stop_probs = self._stop_probs[t] if self._stop_probs is not None and t < len(self._stop_probs) else None

        budget += self._budget_step
        pred, level, power = self._budget_controller.predict_sample(stop_probs=stop_probs,
                                                                    budget=budget,
                                                                    noise=power_noise,
                                                                    current_time=t)
        label = self._labels[t]

        if pred is None:
            is_correct = 0
        else:
            is_correct = float(abs(label - pred) < SMALL_NUMBER)

        self._num_correct.append(is_correct)
        self._target_budgets.append(budget)
        self._levels.append(level)

        # Update the adaptive controller parameters
        if self._system_type == SystemType.ADAPTIVE:
            if pred is not None:
                self._budget_distribution.update(label=pred, level=level, power=power)

            is_end_of_window = (t + 1) % WINDOW_SIZE == 0
            if is_end_of_window:
                self._current_budget = self._budget_distribution.get_budget(t + 1)

            # We only apply the PID controller after the first budget is set. At the beginning, there is little knowledge
            # about the correct budget
            if t >= WINDOW_SIZE - 1:
                power_so_far = self._budget_controller.get_consumed_energy() / (t + 1)

                budget_step = self._pid_controller.step(y_true=self._current_budget, y_pred=power_so_far, time=t)

                if is_end_of_window:
                    self._budget_step = budget_step

    def estimate_validation_results(self, budget: float, max_time: int) -> Tuple[float, float]:
        assert self._controller is not None, 'Must have an internal controller'
        assert isinstance(self._controller, AdaptiveController), 'Can only estimate validation results for adaptive controllers'

        acc, power = self._controller.evaluate(budget=budget, model_results=self._valid_results)

        return acc, power
