import numpy as np
from typing import List
from utils.np_utils import clip_by_norm
from .threshold_optimizer import ThresholdOptimizer


class SimAnnealOptimizer(ThresholdOptimizer):

    def __init__(self, instances: int, epsilon: float, anneal: float, num_candidates: int, move_norm: float, batch_size: int, iterations: int):
        super().__init__(iterations, batch_size)

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

    def init(self, num_features: int) -> List[np.ndarray]:
        states = []
        for _ in range(self.instances - 1):
            init = np.random.uniform(low=0.0, high=1.0, size=(num_features, ))
            states.append(np.sort(init))

        # Always initialize with an all-0.5 distribution
        states.append(np.full(shape=(num_features,), fill_value=0.5))

        return states

    def update(self, state: List[np.ndarray], fitness: List[float], probabilities: np.ndarray, labels: np.ndarray) -> List[np.ndarray]: 
        num_features = len(state[0])
        best_states: List[np.ndarray] = []

        for elem_index, element in enumerate(state):
            moves = np.random.uniform(low=0.0, high=1.0, size=(self.num_candidates, num_features))
            clipped_moves = clip_by_norm(moves, self.move_norm)

            candidates = np.clip(clipped_moves + np.expand_dims(element, axis=0), a_min=0.0, a_max=1.0)
            candidates = np.sort(candidates, axis=-1)
            candidate_fitnesses = self.evaluate(candidates, probabilities, labels)

            best_index = np.argmax(candidate_fitnesses)
            best_candidate = candidates[best_index]

            r = np.random.uniform(low=0.0, high=1.0)
            if candidate_fitnesses[best_index] > fitness[elem_index] or r < self.epsilon:
                best_states.append(best_candidate)
            else:
                best_states.append(element)

        return best_states
