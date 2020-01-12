import numpy as np
from typing import Dict, Any, Optional


class InferencePolicy:

    def __init__(self, max_num_levels: int, name: str):
        self._max_num_levels = max_num_levels
        self._name = name

    @property
    def max_num_levels(self) -> int:
        return self._max_num_levels

    @property
    def name(self) -> str:
        return self._name

    def get_num_levels(self, context: Optional[np.ndarray] = None) -> int:
        raise NotImplementedError()

    def update(self, level: int, reward: float):
        pass

    def reset(self):
        pass


class ConstantPolicy(InferencePolicy):

    def __init__(self, constant: int, max_num_levels: int, name: str):
        super().__init__(max_num_levels, name)
        self._constant = constant

    @property
    def constant(self) -> int:
        return self._constant

    def get_num_levels(self, context: Optional[np.ndarray] = None) -> int:
        return self.constant


class EpsilonGreedyPolicy(InferencePolicy):

    def __init__(self, epsilon: float, max_num_levels: int, name: str):
        super().__init__(max_num_levels, name)
        self._epsilon = epsilon
        self._rewards = np.zeros(shape=(max_num_levels,))
        self._counts = np.ones(shape=(max_num_levels,))  # Initialize to ones to avoid division by zero

    @property
    def epsilon(self) -> float:
        return self._epsilon

    def get_num_levels(self, context: Optional[np.ndarray] = None) -> int:
        if (np.random.random() < self.epsilon):
            return np.random.randint(low=1, high=self.max_num_levels + 1)

        reward_ratios = self._rewards / self._counts
        level = np.argmax(reward_ratios)
        return level + 1

    def update(self, level: int, reward: float):
        self._rewards[level - 1] += reward
        self._counts[level - 1] += 1


def get_inference_policy(name: str, max_num_levels: int, **kwargs: Dict[str, Any]) -> InferencePolicy:
    """
    Factory for inference policies.

    Args:
        name: Name of the inference policy type
        max_num_levels: The maximum number of levels which can be computed
        kwargs: Policy specific parameters
    Returns:
        An inference policy
    """
    name_lower = name.lower()
    if (name == 'constant'):
        return ConstantPolicy(constant=int(kwargs['constant']), max_num_levels=max_num_levels, name=name)
    elif (name == 'epsilon_greedy'):
        return EpsilonGreedyPolicy(epsilon=kwargs['epsilon'], max_num_levels=max_num_levels, name=name)
    raise ValueError(f'Unknown policy {name}.')
