import numpy as np
from typing import Dict, Any


class RechargeEstimator:

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def estimate(self):
        raise NotImplementedError()

    def update(self, measurement: float):
        pass


class ConstantEstimator(RechargeEstimator):

    def __init__(self, constant: float, name: str):
        super().__init__(name)
        self._constant = constant

    @property
    def constant(self) -> float:
        return self._constant

    def estimate(self):
        return self._constant


def get_recharge_estimator(name: str, **kwargs: Dict[str, Any]) -> RechargeEstimator:
    """
    Factory for creating recharge estimators.

    Args:
        name: Name of the estimator type
        kwargs: Dictionary of estimator-specific arguments
    Returns:
        A Recharge Estimator
    """
    name_lower = name.lower()
    if (name_lower == 'constant'):
        return ConstantEstimator(constant=kwargs['constant'], name=name)
    raise ValueError(f'Unknown estimator {name}')
