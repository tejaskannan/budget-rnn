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


class KalmanEstimator(RechargeEstimator):

    def __init__(self,
                 process_var: float,
                 measurement_var: float,
                 init_error_var: float,
                 init_estimate: float,
                 name: str):
        super().__init__(name)
        self._process_var = process_var
        self._measurement_var = measurement_var
        self._error_var = init_error_var
        self._estimate = init_estimate

        self._transition_factor = 1
        self._measurement_factor = 1

    def estimate(self):
        return self._estimate

    def update(self, measurement: float):
        # Compute Kalman gain
        kalman_gain_denominator = self._measurement_factor * self._error_var * self._measurement_factor + self._measurement_var
        kalman_gain = (self._error_var * self._measurement_factor) / kalman_gain_denominator

        # Update error and estimate
        innovation = measurement - self._measurement_factor * self._estimate
        updated_estimate = self._estimate + kalman_gain * innovation
        updated_error_var = (1 - kalman_gain * self._measurement_factor) * self._error_var

        # Transition to next timestep
        self._estimate = self._transition_factor * updated_estimate
        self._error_var = self._transition_factor * updated_error_var * self._transition_factor + self._process_var

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
    elif (name_lower == 'kalman'):
        return KalmanEstimator(process_var=kwargs['process_var'],
                               measurement_var=kwargs['measurement_var'],
                               init_error_var=kwargs['init_error_var'],
                               init_estimate=kwargs['init_estimate'],
                               name=name)
    raise ValueError(f'Unknown estimator {name}')
