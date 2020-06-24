import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score
from typing import Dict, Any, List, Optional, Tuple

from dataset.dataset import Dataset, DataSeries
from models.adaptive_model import AdaptiveModel
from utils.rnn_utils import get_logits_name
from utils.testing_utils import ClassificationMetric
from utils.np_utils import round_to_precision, min_max_normalize, clip_by_norm
from utils.constants import BIG_NUMBER, SMALL_NUMBER, OUTPUT
from utils.adaptive_inference import threshold_predictions, optimal_levels, normalize_logits


class ThresholdOptimizer:
    
    def __init__(self, params: Dict[str, Any], model: AdaptiveModel):
        self._model = model
        self._thresholds = None
        self._num_levels = model.num_outputs
        self._level_penalty = params.get('level_penalty', 0.0)

        # Power Estimates from profiling (constant for now)
        self._avg_power = np.array([24.085, 32.776, 37.897, 43.952, 48.833, 50.489, 54.710, 57.692, 59.212, 59.251])

    def fit(self, dataset: Dataset, series: DataSeries):
        raise NotImplementedError()

    def identifier(self) -> Tuple[Any, Any]:
        raise NotImplementedError()

    def score(self, dataset: Dataset, series: DataSeries, flops_per_level: List[float], thresholds: Optional[np.ndarray] = None) -> Dict[str, Any]:
        assert self._thresholds is not None or thresholds is not None, 'Must fit the model or provide thresholds.'

        if thresholds is None:
            thresholds = self._thresholds

        # Set logit operations
        logit_ops = [get_logits_name(i) for i in range(self._num_levels)]

        # Converts flops list to numpy array
        flops_per_level = np.array(flops_per_level)

        data_generator = dataset.minibatch_generator(series=series,
                                                     batch_size=self._batch_size,
                                                     metadata=self._model.metadata,
                                                     should_shuffle=False)
 
        predictions: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        levels: List[np.ndarray] = []
        flops: List[np.ndarray] = []

        for batch in data_generator:
            # Compute the predicted log probabilities
            feed_dict = self._model.batch_to_feed_dict(batch, is_train=False)
            logits = self._model.execute(feed_dict, logit_ops)

            # Concatenate logits into a [B, L, C] array (logit_ops is already ordered by level).
            # For reference, L is the number of levels and C is the number of classes
            logits_concat = np.concatenate([np.expand_dims(logits[op], axis=1) for op in logit_ops], axis=1)

            # Normalize logits and round to fixed point representation
            normalized_logits = normalize_logits(logits_concat, precision=self._precision)

            # [S, L]
            batch_predictions, batch_levels = threshold_predictions(normalized_logits, thresholds=thresholds)
            batch_labels = np.squeeze(batch[OUTPUT])

            predictions.append(batch_predictions)
            levels.append(batch_levels)
            labels.append(batch_labels)

            batch_flops = flops_per_level[batch_levels]
            flops.append(batch_flops)

        predictions = np.concatenate(predictions, axis=-1)
        labels = np.concatenate(labels, axis=-1)
        levels = np.concatenate(levels, axis=-1)
        flops = np.concatenate(flops, axis=-1)

        # Compute the (approx) average power
        level_distribution = np.vstack([np.bincount(levels[i, :], minlength=self._num_levels) for i in range(levels.shape[0])])  # [S, L]
        level_distribution = level_distribution / np.sum(level_distribution, axis=-1, keepdims=True)  # [S, L]
        approx_power = np.sum(level_distribution * self._avg_power, axis=-1)  # [S]

        return {
            ClassificationMetric.ACCURACY.name: np.average((labels == predictions).astype(float), axis=-1).tolist(),
            ClassificationMetric.MACRO_F1_SCORE.name: [f1_score(labels, predictions[i, :], average='macro') for i in range(predictions.shape[0])],
            ClassificationMetric.MICRO_F1_SCORE.name: [f1_score(labels, predictions[i, :], average='micro') for i in range(predictions.shape[0])],
            ClassificationMetric.LEVEL.name: np.average(levels + 1, axis=-1),
            ClassificationMetric.FLOPS.name: np.average(flops, axis=-1),
            'THRESHOLDS': thresholds.astype(float).tolist(),
            'APPROX_POWER': approx_power.astype(float)
        }

    def fitness_function(self, normalized_logits: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, penalty: Optional[float] = None) -> float:
        predictions, levels = threshold_predictions(normalized_logits, thresholds=thresholds)

        assert predictions.shape == labels.shape, 'Misaligned labels ({0}) and predictions ({1})'.format(labels.shape, predictions.shape)

        penalty_factor = penalty if penalty is not None else self._level_penalty

        accuracy = np.average((predictions == labels).astype(float))

        level_diff = levels - optimal_levels(normalized_logits, labels)
        clipped_diff = np.clip(level_diff, a_min=0.0, a_max=None)
        level_penalty = -1 * penalty_factor * (np.sum(clipped_diff) / (np.sum(np.where(level_diff >= 0, 1, 0) + SMALL_NUMBER)))

        return accuracy + level_penalty


