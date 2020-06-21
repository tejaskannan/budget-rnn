import numpy as np
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Tuple

from dataset.dataset import Dataset, DataSeries
from models.adaptive_model import AdaptiveModel
from utils.rnn_utils import get_logits_name
from utils.testing_utils import ClassificationMetric
from utils.np_utils import round_to_precision, min_max_normalize, clip_by_norm
from utils.constants import BIG_NUMBER, SMALL_NUMBER, OUTPUT
from utils.adaptive_inference import threshold_predictions
from threshold_optimization.optimizer import ThresholdOptimizer


class BudgetType(Enum):
    POWER = auto()  # Maximize accuracy subject to a power constraint
    ACCURACY = auto()  # Minimize power subject to an accuracy constraint


class GreedyThresholdOptimizer(ThresholdOptimizer):

    def __init__(self, params: Dict[str, Any], model: AdaptiveModel):
        super().__init__(params, model)
        
        self._precision = params['precision']
        self._batch_size = params.get('batch_size', model.hypers.batch_size)
        # self._tolerance = params['tolerance']
        self._should_sort_thresholds = params['should_sort_thresholds']
        # self._should_anneal_penalty = params['anneal_penalty']
        self._trials = params['trials']
        
        # Power Estimates from profiling (constant for now)
        self._avg_power = np.array([24.085, 32.776, 37.897, 43.952, 48.833, 50.489, 54.710, 57.692, 59.212, 59.251])

        # Constraints during optimization
        self._budget_type = BudgetType[params['budget_type'].upper()]
        self._budget = params['budget_value']
        self._violation_factor = 100 if self._budget_type == BudgetType.POWER else np.max(self._avg_power) + 100

    @property
    def identifier(self) -> Tuple[Any, Any]:
        return (self._budget_type.name.lower(), self._budget)

    def fitness_function(self, normalized_logits: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, penalty: Optional[float] = None) -> float:
        predictions, levels = threshold_predictions(normalized_logits, thresholds=thresholds)

        assert predictions.shape == labels.shape, 'Misaligned labels ({0}) and predictions ({1})'.format(labels.shape, predictions.shape)

        penalty_factor = penalty if penalty is not None else self._level_penalty

        accuracy = np.average((predictions == labels).astype(float))

        level_counts = np.bincount(levels, minlength=self._num_levels)  # [L]
        normalized_level_counts = level_counts / np.sum(level_counts)
        approx_power = np.sum(normalized_level_counts * self._avg_power).astype(float)

        # Determine the objective function based on the budget type
        dual_penalty, fitness = 0.0, 0.0
        if self._budget_type == BudgetType.ACCURACY:
            dual_penalty = 100 * (self._budget - accuracy)  # Convert to percentage to match the scale of power values
            fitness = approx_power
        elif self._budget_type == BudgetType.POWER:
            dual_penalty = approx_power - self._budget
            fitness = -accuracy
        else:
            raise ValueError('Unknown budget type: {0}'.format(self._budget_type))

        return fitness + self._violation_factor * np.clip(dual_penalty, a_min=0.0, a_max=None)


    def fit(self, dataset: Dataset, series: DataSeries):
        # Set logit operations
        logit_ops = [get_logits_name(i) for i in range(self._num_levels)]

        # Compute all logits. TODO: Use large batches i.e. 1024 and do this optimization incrementally.
        data_generator = dataset.minibatch_generator(series=series,
                                                     batch_size=self._batch_size,
                                                     metadata=self._model.metadata,
                                                     should_shuffle=False)

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
            normalized_batch_logits = min_max_normalize(logits_concat, axis=-1)
            normalized_batch_logits = round_to_precision(normalized_batch_logits, precision=self._precision)

            normalized_logits.append(normalized_batch_logits)
            labels.append(np.squeeze(batch[OUTPUT]))

        normalized_logits = np.concatenate(normalized_logits, axis=0)
        labels = np.concatenate(labels, axis=0)

        # Multiplier for fixed point conversion
        fp_one = 1 << self._precision

        # Greedily optimize each threshold via an exhaustive search over all fixed point values
        thresholds = np.ones(shape=(self._num_levels, ))

        # Track previous thresholds to detect convergence
        prev_thresholds = np.copy(thresholds)

        for trial in range(self._trials):
            print('====== Starting Trial {0} ======'.format(trial))

            for level in range(self._num_levels - 1):
                
                # Determine threshold ranges
                if self._should_sort_thresholds:
                    max_threshold = int(thresholds[level - 1] * fp_one) if level > 0 else fp_one
                else:
                    max_threshold = fp_one

                best_t, best_obj = None, None
                for t in range(max_threshold + 1):
                    thresholds[level] = t / fp_one
                    objective = self.fitness_function(normalized_logits, labels, thresholds=thresholds, penalty=self._level_penalty)

                    if best_obj is None or objective < best_obj:
                        best_obj = objective
                        best_t = t / fp_one

                
                objective_value = best_obj if self._budget_type == BudgetType.ACCURACY else -best_obj
                print('Completed level {0}. Objective: {1:.4f}'.format(level + 1, objective_value))

                thresholds[level] = best_t

            print('Finished Trial {0}. Objective: {1:.4f}. Thresholds: {2}'.format(trial + 1, objective_value, thresholds))

            if np.isclose(prev_thresholds, thresholds).all():
                print('Converged.')
                break

            prev_thresholds = np.copy(thresholds)

        self._thresholds = thresholds
        return thresholds
