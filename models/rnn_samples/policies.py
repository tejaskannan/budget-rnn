import numpy as np


class InferencePolicy:

    def __init__(self, max_num_levels: int):
        self._max_num_levels = max_num_levels

    @property
    def max_num_levels(self):
        return self._max_num_levels

    def get_num_levels(self):
        return self.max_num_levels

    def update(self, reward: float):
        pass


class RechargeEstimator:

    def estimate(self):
        return 5.0

    def update(self, measurement: float):
        pass
