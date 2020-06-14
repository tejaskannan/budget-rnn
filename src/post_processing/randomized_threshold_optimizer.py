import numpy as np
from collections import namedtuple
from typing import List, Dict, Iterable, Any, Optional

from dataset.dataset import DataSeries, Dataset
from models.adaptive_model import AdaptiveModel
from utils.rnn_utils import get_logits_name
from utils.constants import SMALL_NUMBER, OUTPUT
from utils.np_utils import f1_score, softmax, sigmoid, linear_normalize, min_max_normalize, multiclass_f1_score
from utils.threshold_utils import adaptive_inference, TwoSidedThreshold, InferenceMode

from .threshold_optimizer import ThresholdOptimizer, OptimizerOutput


class RandomizedThresholdOptimizer(ThresholdOptimizer):
    """
    Optimizes probability thresholds using a genetic algorithm.
    """

    def __init__(self, iterations: int, batch_size: int, level_weight: float, mode: str):
        super().__init__(iterations, batch_size, mode)
        self._level_weight = level_weight

    @property
    def level_weight(self) -> float:
        return self._level_weight

    def optimize(self, model: AdaptiveModel, dataset: Dataset) -> OptimizerOutput:
        """
        Runs the genetic algorithm optimization.
        """
        # Create data iterator to compute fitness
        data_generator = self.get_data_generator(dataset, model.metadata)

        # Initialize the state
        state = self.init(model.num_outputs)

        # Set logit operations
        logit_ops = [get_logits_name(i) for i in range(model.num_outputs)]

        best_score = 0.0
        best_thresholds = None

        # Compute optimization steps per batch
        batch = next(data_generator)
        for batch_num in range(self.iterations):
            feed_dict = model.batch_to_feed_dict(batch, is_train=False)
            logits = model.execute(feed_dict, logit_ops)

            # Concatenate logits into a 2D array (logit_ops is already ordered by level)
            logits_concat = np.concatenate([logits[op] for op in logit_ops], axis=-1)
            
            if self.inference_mode == InferenceMode.MULTICLASS:
                logits_concat = np.concatenate([np.expand_dims(logits[op], axis=1) for op in logit_ops], axis=1)
                probabilities = min_max_normalize(logits_concat, axis=-1)
                # probabilities = softmax(logits_concat, axis=-1)
            else:
                logits_concat = np.concatenate([logits[op] for op in logit_ops], axis=-1)
                probabilities = sigmoid(logits_concat)
 
            labels = np.squeeze(np.vstack(batch[OUTPUT]), axis=-1)

            fitness = self.evaluate(state, probabilities, labels)

            # Avoid cases in which all labels are zero (both true positive and false negative rates will be zero)
            if any([abs(x) > SMALL_NUMBER for x in labels]):
                best_index = np.argmax(fitness)
                best_score = fitness[best_index]
                best_thresholds = state[best_index]

                state = self.update(state, fitness, probabilities, labels)

            if self.has_converged(state):
                print(f'Converged in {batch_num + 1} iterations. Score so far: {best_score:.3f}.', end='\r')
                break

            try:
                batch = next(data_generator)
            except StopIteration:
                data_generator = self.get_data_generator(dataset, model.metadata)
                batch = next(data_generator)

            print(f'Completed {batch_num + 1} / {self.iterations} iterations. Score so far: {best_score:.3f}.', end='\r')

        print()

        return OptimizerOutput(score=best_score, thresholds=best_thresholds)

    def init(self, num_features: int) -> List[Any]:
        raise NotImplementedError()

    def has_converged(self, state: List[Any]) -> bool:
        state_array = np.array(state)

        if self.inference_mode == InferenceMode.BINARY_ONE_SIDED:
            lower = np.array([t.lower for t in state[0]])

            return np.isclose(state_array[:, :, 0], lower).all()
        else:
            return np.isclose(state_array, state_array[0]).all()

    def update(self, state: List[Any], fitness: List[float], probabilities: np.ndarray, labels: np.ndarray) -> List[Any]:
        raise NotImplementedError()

    def evaluate(self, state: List[Any], probabilities: np.ndarray, labels: np.ndarray) -> List[float]:
        fitnesses: List[float] = []

        for element in state:
            output = adaptive_inference(probabilities, element, mode=self.inference_mode)
            predictions = output.predictions
            levels = output.indices

            num_levels = probabilities.shape[1]
            level_penalty = self.level_weight * np.average(levels / num_levels)

            if self.inference_mode == InferenceMode.MULTICLASS:
                fitness = multiclass_f1_score(predictions, labels, num_classes=self.num_classes) - level_penalty
            else:
                fitness = f1_score(predictions, labels) - level_penalty

            fitnesses.append(fitness)

        return fitnesses
