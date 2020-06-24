import numpy as np
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Tuple

from dataset.dataset import Dataset, DataSeries
from models.adaptive_model import AdaptiveModel
from utils.rnn_utils import get_logits_name
from utils.testing_utils import ClassificationMetric
from utils.np_utils import round_to_precision, min_max_normalize, clip_by_norm
from utils.constants import BIG_NUMBER, SMALL_NUMBER, OUTPUT
from utils.adaptive_inference import threshold_predictions, normalize_logits
from threshold_optimization.optimizer import ThresholdOptimizer


class BudgetType(Enum):
    POWER = auto()  # Maximize accuracy subject to a power constraint
    ACCURACY = auto()  # Minimize power subject to an accuracy constraint


class GreedyThresholdOptimizer(ThresholdOptimizer):

    def __init__(self, params: Dict[str, Any], model: AdaptiveModel):
        super().__init__(params, model)
        
        self._precision = params['precision']
        self._batch_size = params.get('batch_size', model.hypers.batch_size)
        self._should_sort_thresholds = params['should_sort_thresholds']
        self._trials = params['trials']

        # Constraints during optimization
        self._budget_type = BudgetType[params['budget_type'].upper()]
        self._budgets = params['budget_values']
        self._violation_factor = 1000 if self._budget_type == BudgetType.POWER else np.max(self._avg_power) + 1000

        # Convert the budgets into a [S] numpy array
        if not isinstance(self._budgets, list):
            self._budgets = [self._budgets]
        self._budgets = np.array(self._budgets)


    @property
    def identifier(self) -> Tuple[Any, Any]:
        return (self._budget_type.name.lower(), self._budgets)

    def fitness_function(self, normalized_logits: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, penalty: Optional[float] = None) -> np.ndarray:
        predictions, levels = threshold_predictions(normalized_logits, thresholds=thresholds)  # Pair of [S, B] arrays

        labels = np.expand_dims(labels, axis=0)
        assert predictions.shape[-1] == labels.shape[-1], 'Misaligned labels ({0}) and predictions ({1})'.format(labels.shape, predictions.shape)

        penalty_factor = penalty if penalty is not None else self._level_penalty

        accuracy = np.average((predictions == labels).astype(float), axis=-1)  # [S]

        level_counts = np.vstack([np.bincount(levels[i, :], minlength=self._num_levels) for i in range(levels.shape[0])])  # [S, L]
        normalized_level_counts = level_counts / np.sum(level_counts, axis=-1, keepdims=True)  # [S, L]
        approx_power = np.sum(normalized_level_counts * self._avg_power, axis=-1).astype(float)  # [S]

        # Determine the objective function based on the budget type
        dual_penalty, fitness = 0.0, 0.0
        if self._budget_type == BudgetType.ACCURACY:
            dual_penalty = 100 * (self._budget - accuracy)  # Convert to percentage to match the scale of power values
            fitness = approx_power
        elif self._budget_type == BudgetType.POWER:
            dual_penalty = approx_power - self._budgets  # [S]
            fitness = -accuracy  # [S]
        else:
            raise ValueError('Unknown budget type: {0}'.format(self._budget_type))

        dual_penalty = np.where(dual_penalty < 0, 0.0, self._violation_factor * dual_penalty)

        return fitness + dual_penalty
        # return fitness + self._violation_factor * np.clip(dual_penalty, a_min=0.0, a_max=None)  # [S]


    def fit(self, dataset: Dataset, series: DataSeries):
        # Set logit operations
        logit_ops = [get_logits_name(i) for i in range(self._num_levels)]

        # Compute all logits. TODO: Use large batches i.e. 1024 and do this optimization incrementally.
        data_generator = dataset.minibatch_generator(series=series,
                                                     batch_size=self._batch_size,
                                                     metadata=self._model.metadata,
                                                     should_shuffle=False,
                                                     drop_incomplete_batches=False)

        normalized_logits: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        for batch in data_generator:
            # Compute the predicted log probabilities
            feed_dict = self._model.batch_to_feed_dict(batch, is_train=False)
            logits = self._model.execute(feed_dict, logit_ops)

            # Concatenate logits into a [B, L, C] array (logit_ops is already ordered by level).
            # For reference, L is the number of levels and C is the number of classes
            logits_concat = np.concatenate([np.expand_dims(logits[op], axis=1) for op in logit_ops], axis=1)

            # Normalize logits and round to fixed point representation
            normalized_batch_logits = normalize_logits(logits_concat, precision=self._precision)

            normalized_logits.append(normalized_batch_logits)
            labels.append(np.squeeze(batch[OUTPUT]))

        normalized_logits = np.concatenate(normalized_logits, axis=0)
        labels = np.concatenate(labels, axis=0)

        # Multiplier for fixed point conversion
        fp_one = 1 << self._precision

        # Greedily optimize each threshold via an exhaustive search over all fixed point values
        thresholds = np.ones(shape=(self._budgets.shape[0], self._num_levels))
        # thresholds = np.full(shape=(self._num_levels, ), fill_value=0.5)
        thresholds[:, -1] = 0

        # Track previous thresholds to detect convergence
        prev_thresholds = np.copy(thresholds)

        for trial in range(self._trials):
            print('====== Starting Trial {0} ======'.format(trial))

            for level in reversed(range(self._num_levels - 1)):

                # 1 when constraint violated at this level and 0 when satisfied
                power_comparison = (self._avg_power[level] > self._budgets).astype(float)
                if trial == 0 and (power_comparison > SMALL_NUMBER).all():
                    continue

                # Determine threshold ranges
                if self._should_sort_thresholds and trial > 0:
                    min_threshold = int(thresholds[level + 1] * fp_one)
                else:
                    min_threshold = 0

                best_t = np.ones_like(self._budgets)
                best_obj = np.zeros_like(self._budgets) + BIG_NUMBER

                for t in range(min_threshold, fp_one, 1):
                    thresholds[:, level] = t / fp_one
                    objectives = self.fitness_function(normalized_logits, labels, thresholds=thresholds, penalty=self._level_penalty)  # [S]

                    # At the first trial, apply the power comparison mask to avoid wrongly setting values
                    if trial == 0:
                        objectives = (power_comparison * BIG_NUMBER) + (1.0 - power_comparison) * objectives

                    best_t = np.where(objectives < best_obj, t / fp_one, best_t)
                    best_obj = np.minimum(objectives, best_obj)

                   # if best_obj is None or objective < best_obj:
                   #     best_obj = objective
                   #     best_t = t / fp_one

                objective_value = best_obj if self._budget_type == BudgetType.ACCURACY else -best_obj
                print('Completed level {0}. Objectives: {1}'.format(level + 1, objective_value))

                thresholds[:, level] = best_t

            print('Finished Trial {0}. Objectives: {1}. Thresholds: {2}'.format(trial + 1, objective_value, thresholds))

            if np.isclose(prev_thresholds, thresholds).all():
                print('Converged.')
                break

            prev_thresholds = np.copy(thresholds)

        self._thresholds = thresholds
        return thresholds
