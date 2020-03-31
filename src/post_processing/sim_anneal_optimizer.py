import numpy as np
from typing import List

from utils.np_utils import clip_by_norm
from utils.threshold_utils import TwoSidedThreshold, order_threshold_lists, matrix_to_thresholds
from utils.constants import ONE_HALF

from .randomized_threshold_optimizer import RandomizedThresholdOptimizer


LOWER_BOUND = 0.0
UPPER_BOUND = 1.0


class SimAnnealOptimizer(RandomizedThresholdOptimizer):

    def __init__(self, instances: int, epsilon: float, anneal: float, num_candidates: int, move_norm: float, batch_size: int, iterations: int, level_weight: float, mode: str):
        super().__init__(iterations, batch_size, level_weight, mode)

        assert epsilon >= 0.0 and epsilon <= 1.0, 'The epsilon value must be in [0, 1]'
        assert anneal > 0 and anneal < 1, 'The anneal value must be in (0, 1)'

        self._instances = instances
        self._epsilon = epsilon
        self._anneal = anneal
        self._num_candidates = num_candidates
        self._move_norm = move_norm

    @property
    def instances(self) -> int:
        return self._instances

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def anneal(self) -> float:
        return self._anneal

    @property
    def num_candidates(self) -> int:
        return self._num_candidates

    @property
    def move_norm(self) -> float:
        return self._move_norm

    def anneal_epsilon(self):
        self._epsilon *= self.anneal

    def init(self, num_features: int) -> List[TwoSidedThreshold]:
        states = []
        for _ in range(self.instances - 1):
            init = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=(num_features, 2))
            thresholds = [TwoSidedThreshold(lower=np.min(x), upper=np.max(x)) for x in init]
            states.append(thresholds)

        # Always initialize with an all-0.5 distribution
        thresholds = [TwoSidedThreshold(lower=0.5, upper=1.0) for _  in range(num_features)]
        states.append(thresholds)

        return states

    def update(self, state: List[TwoSidedThreshold], fitness: List[float], probabilities: np.ndarray, labels: np.ndarray) -> List[TwoSidedThreshold]:
        num_features = len(state[0])
        best_states: List[List[TwoSidedThreshold]] = []

        for elem_index, element in enumerate(state):
            # Generate random moves
            lower_moves = np.random.normal(loc=0.0, scale=1.0, size=(self.num_candidates, num_features, ))  # [K, L]
            upper_moves = np.random.normal(loc=0.0, scale=1.0, size=(self.num_candidates, num_features, ))  # [K, L]

            # Clip moves
            clipped_lower = np.expand_dims(clip_by_norm(lower_moves, self.move_norm), axis=-1)  # [K, L, 1]
            clipped_upper = np.expand_dims(clip_by_norm(upper_moves, self.move_norm), axis=-1)  # [K, L, 1]

            # Concatenate clipped moves into a [K, L, 2] matrix
            clipped_moves = np.concatenate([clipped_lower, clipped_upper], axis=-1)

            element_matrix = np.expand_dims(np.array(element), axis=0)  # [1, L, 2]

            # Form candidates, [K, L, 2]
            candidates = np.clip(clipped_moves + element_matrix, a_min=LOWER_BOUND, a_max=UPPER_BOUND)
            
            # Convert back to a list of thresholds
            candidate_thresholds = list(map(matrix_to_thresholds, candidates))

            # Evaluate the formed candidates
            candidate_fitnesses = self.evaluate(candidate_thresholds, probabilities, labels)

            # Select the best candidates
            best_index = np.argmax(candidate_fitnesses)
            best_candidate = candidate_thresholds[best_index]

            r = np.random.uniform(low=0.0, high=1.0)
            if candidate_fitnesses[best_index] > fitness[elem_index] or r < self.epsilon:
                best_states.append(best_candidate)
            else:
                best_states.append(element)

        # Anneal the random parameter
        self.anneal_epsilon()

        return best_states
