import numpy as np
import os.path
from collections import namedtuple
from enum import Enum, auto
from typing import List, Dict, Any

from controllers.controller_utils import clip, get_budget_index, ModelResults
from controllers.model_controllers import Controller, FixedController, RandomController, BudgetWrapper, CONTROLLER_PATH
from controllers.model_controllers import SkipRNNController
from controllers.runtime_controllers import PIDController, BudgetController, BudgetDistribution
from dataset.dataset import DataSeries, Dataset
from models.adaptive_model import AdaptiveModel
from models.tf_model import TFModel
from utils.rnn_utils import get_logits_name, get_stop_output_name
from utils.file_utils import extract_model_name
from utils.constants import INPUTS, OUTPUT, SEQ_LENGTH, NUM_CLASSES, SEQ_LENGTH, SMALL_NUMBER


# Constants
KP = 1.0
KD = 1.0 / 32.0
KI = 1.0 / 64.0
WINDOW_SIZE = 20
INTEGRAL_BOUNDS = (-100, 100)
PANIC_FRAC = 0.95


# Defines the type of runtime system
class SystemType(Enum):
    ADAPTIVE = auto()
    RANDOMIZED = auto()
    GREEDY = auto()
    FIXED = auto()
    SKIP_RNN = auto()


def estimate_label_counts(controller: Controller, budget: float, num_levels: int, num_classes: int) -> Dict[int, np.ndarray]:
    # TODO: This operation is expensive because it executes the model on the validation set. Instead, we should give the validation stop outputs
    # and use the controller on the already_computed results. This requires executing the adaptive models on the validation set AND the testing set
    # before starting any simulations.
    levels, predictions = controller.predict_levels(series=DataSeries.VALID, budget=budget)
    batch_size = predictions.shape[0]

    result: Dict[int, np.ndarray] = dict()

    # Estimate power for each class
    for class_idx in range(num_classes):
        class_mask = (predictions == class_idx).astype(int)

        class_level_counts = np.bincount(levels, minlength=num_levels, weights=class_mask)
        result[class_idx] = class_level_counts

    return result


class RuntimeSystem:

    def __init__(self, model_results: ModelResults, system_type: str, model_path: str, dataset_folder: str, num_classes: int, num_levels: int, seq_length: int):
        self._system_type = SystemType[system_type.upper()]
        self._model_results = model_results

        self._model_path = model_path
        self._dataset_folder = dataset_folder

        self._num_classes = num_classes
        self._num_levels = num_levels
        self._seq_length = seq_length

        save_folder, model_file_name = os.path.split(model_path)
        model_name = extract_model_name(model_file_name)
        
        self._model_name = model_name
        self._save_folder = save_folder
            
        # If the system is adaptive, we can load the controller now. Otherwise, we wait until later
        if self._system_type == SystemType.ADAPTIVE:
            self._controller = Controller.load(os.path.join(save_folder, CONTROLLER_PATH.format(model_name)), dataset_folder=dataset_folder)
        elif self._system_type == SystemType.SKIP_RNN:
            self._controller = SkipRNNController(sample_counts=model_results.stop_probs,
                                                 seq_length=seq_length)
        else:
            self._controller = None

        self._level_predictions = model_results.predictions
        self._stop_probs = model_results.stop_probs
        self._level_accuracy = model_results.accuracy
        self._labels = model_results.labels
        self._budget_controller = None

        self._name = '{0} {1}'.format(model_name.split('-')[0], system_type.upper())
        self._seed = hash(self._name) % 1000

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
        return np.array(self._budget_controller.power)

    def get_energy(self) -> np.ndarray:
        return np.cumsum(self._budget_controller.power)

    def get_num_correct(self) -> np.ndarray:
        return np.cumsum(self._num_correct)

    def get_target_budgets(self) -> np.ndarray:
        return np.array(self._target_budgets).reshape(-1)

    def init_for_budget(self, budget: float, max_time: int):
        # Make controller based on the model type
        if self._system_type == SystemType.RANDOMIZED:
            self._controller = RandomController(budgets=[budget], seq_length=self._seq_length, num_levels=self._num_levels)
            self._controller.fit(series=None)
        elif self._system_type == SystemType.GREEDY:
            level = np.argmax(self._level_accuracy)
            self._controller = FixedController(model_index=level)
        elif self._system_type == SystemType.FIXED:
            level = get_budget_index(budget=budget, level_accuracy=self._level_accuracy)
            self._controller = FixedController(model_index=level)
        elif self._system_type == SystemType.ADAPTIVE:
            # Make the budget distribution and PID controller
            output_range = (0, self._num_levels - 1)
            self._pid_controller = BudgetController(kp=KP,
                                                    ki=KI,
                                                    kd=KD,
                                                    integral_bounds=INTEGRAL_BOUNDS,
                                                    output_range=output_range,
                                                    budget=budget,
                                                    window=WINDOW_SIZE)
            # Create the power distribution
            prior_counts = estimate_label_counts(self._controller, budget, self._num_levels, self._num_classes)
            self._budget_distribution = BudgetDistribution(prior_counts=prior_counts,
                                                           budget=budget,
                                                           max_time=max_time,
                                                           num_levels=self._num_levels,
                                                           num_classes=self._num_classes,
                                                           seq_length=self._seq_length,
                                                           panic_frac=PANIC_FRAC)

        # Apply budget wrapper to the controller
        assert self._controller is not None, 'Must have a valid controller'
        self._budget_controller = BudgetWrapper(controller=self._controller,
                                                model_predictions=self._level_predictions,
                                                max_time=max_time,
                                                num_classes=self._num_classes,
                                                num_levels=self._num_levels,
                                                seq_length=self._seq_length,
                                                budget=budget,
                                                seed=self._seed)
        self._budget_step = 0
        self._num_correct = []
        self._target_budgets = []

    def step(self, budget: float, power_noise: float, time: int):
        stop_probs = self._stop_probs[time] if self._stop_probs is not None and time < len(self._stop_probs) else None

        budget += self._budget_step
        pred, level = self._budget_controller.predict_sample(stop_probs=stop_probs,
                                                             budget=budget,
                                                             noise=power_noise,
                                                             current_time=time)
        label = self._labels[time]
        is_correct = float(abs(label - pred) < SMALL_NUMBER)

        self._num_correct.append(is_correct)
        self._target_budgets.append(budget)

        # Update the adaptive controller parameters
        if time % WINDOW_SIZE == 0 and self._system_type == SystemType.ADAPTIVE:
            current_budget = self._budget_distribution.get_budget(time + 1)
            power_so_far = np.average(self._budget_controller.power)
            self._budget_step = self._pid_controller.step(y_true=current_budget, y_pred=power_so_far, time=time)
