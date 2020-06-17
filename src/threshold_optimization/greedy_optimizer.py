import numpy as np
from typing import Dict, Any, Optional, List

from dataset.dataset import Dataset, DataSeries
from models.adaptive_model import AdaptiveModel
from utils.rnn_utils import get_logits_name
from utils.testing_utils import ClassificationMetric
from utils.np_utils import round_to_precision, min_max_normalize, clip_by_norm
from utils.constants import BIG_NUMBER, SMALL_NUMBER, OUTPUT
from utils.adaptive_inference import threshold_predictions
from threshold_optimization.optimizer import ThresholdOptimizer


class GreedyThresholdOptimizer(ThresholdOptimizer):

    def __init__(self, params: Dict[str, Any], model: AdaptiveModel):
        super().__init__(params, model)
        
        self._precision = params['precision']
        self._batch_size = params.get('batch_size', model.hypers.batch_size)
        self._tolerance = params['tolerance']
        self._should_sort_thresholds = params['should_sort_thresholds']
        self._should_anneal_penalty = params['anneal_penalty']
        self._trials = params['trials']

    def fit(self, dataset: Dataset, series: DataSeries):
        # Set logit operations
        logit_ops = [get_logits_name(i) for i in range(self._num_levels)]

        # Compute all logits
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

        # Compute the penalties for each trial
        if self._should_anneal_penalty and self._trials > 1:
            penalties = np.linspace(start=0.0, stop=self._level_penalty, endpoint=True, num=self._trials)
        else:
            penalties = np.full(shape=(self._trials, ), fill_value=self._level_penalty)

        prev_thresholds = np.copy(thresholds)
        has_changed = np.array([True for _ in thresholds])
        for trial in range(self._trials):
            level_penalty = penalties[trial]

            for level in range(self._num_levels - 1):
                print('Starting level {0}'.format(level + 1))

                if not has_changed[0:level + 1].any():
                    continue

                if self._should_sort_thresholds:
                    max_threshold = int(thresholds[level - 1] * fp_one) if level > 0 else fp_one
                else:
                    max_threshold = fp_one

                best_t, best_obj = None, None

                for t in range(max_threshold + 1):
                    thresholds[level] = t / fp_one
                    objective = self.fitness_function(normalized_logits, labels, thresholds=thresholds, penalty=level_penalty)

                    if best_obj is None or objective >= best_obj:
                        best_obj = objective
                        best_t = t / fp_one

                thresholds[level] = best_t
                has_changed[level] = np.logical_not(np.isclose(prev_thresholds[level], thresholds[level]))

            print('Finished Trial {0}. Thresholds: {1}'.format(trial + 1, thresholds))

            if not self._should_anneal_penalty and not has_changed.any():
                print('Converged.')
                break

            prev_thresholds = np.copy(thresholds)

        self._thresholds = thresholds
        return thresholds
