import numpy as np
from typing import Dict, Any, Optional


class InferencePolicy:

    def __init__(self, inference_period: float, max_num_levels: int, name: str):
        self._max_num_levels = max_num_levels
        self._name = name
        self._inference_period = inference_period

    @property
    def max_num_levels(self) -> int:
        return self._max_num_levels

    @property
    def name(self) -> str:
        return self._name

    @property
    def inference_period(self) -> float:
        return self._inference_period

    def get_num_levels(self, context: Optional[np.ndarray] = None) -> int:
        raise NotImplementedError()

    def update(self, level: int, computed_levels: int, period_time: float):
        pass

    def reset(self):
        pass


class ConstantPolicy(InferencePolicy):

    def __init__(self, constant: int, inference_period: float, max_num_levels: int, name: str):
        super().__init__(inference_period, max_num_levels, name)
        self._constant = constant

    @property
    def constant(self) -> int:
        return self._constant

    def get_num_levels(self, context: Optional[np.ndarray] = None) -> int:
        return self.constant


class EpsilonGreedyPolicy(InferencePolicy):

    def __init__(self, epsilon: float, inference_period: float, max_num_levels: int, name: str):
        super().__init__(inference_period, max_num_levels, name)
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

    def _get_reward(self, level: int, computed_levels: int, period_time: float):
        time_delta = np.abs(self.inference_period - period_time)
        level_delta = np.abs(level - computed_levels)
        return -1 * (time_delta + level_delta)

    def update(self, level: int, computed_levels: int, period_time: float):
        self._rewards[level - 1] += self._get_reward(level, computed_levels, period_time)
        self._counts[level - 1] += 1


def get_inference_policy(name: str, inference_period: float, max_num_levels: int, **kwargs: Dict[str, Any]) -> InferencePolicy:
    """
    Factory for inference policies.

    Args:
        name: Name of the inference policy type
        inference_period: Desired inference period
        max_num_levels: The maximum number of levels which can be computed
        kwargs: Policy specific parameters
    Returns:
        An inference policy
    """
    name_lower = name.lower()
    if (name == 'constant'):
        return ConstantPolicy(constant=int(kwargs['constant']), inference_period=inference_period, max_num_levels=max_num_levels, name=name)
    elif (name == 'epsilon_greedy'):
        return EpsilonGreedyPolicy(epsilon=kwargs['epsilon'], inference_period=inference_period, max_num_levels=max_num_levels, name=name)
    raise ValueError(f'Unknown policy {name}.')
