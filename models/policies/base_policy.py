from typing import Dict, Any


class Policy:

    def __init__(self, ops_per_sec: float, name: str):
        assert ops_per_sec > 0.0, f'The operations per second must be positive'

        self._name = name
        self._ops_per_sec = ops_per_sec
        self._prev_time = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def ops_per_sec(self) -> float:
        return self._ops_per_sec

    @property
    def sec_per_op(self) -> float:
        return 1.0 / self._ops_per_sec

    def update(self, time: float):
        self._prev_time = time

    def should_continue(self, prediction_level: int, time_so_far: float) -> bool:
        """
        Returns True if the anytime model should continue. False otherwise.
        """
        raise NotImplementedError()


class NoPredictionPolicy(Policy):

    def should_continue(self, prediction_level: int, time_so_far: float) -> bool:
        return time_so_far < self.sec_per_op


class PreviousMeasurementPolicy(Policy):
   
    def should_continue(self, prediction_level: int, time_so_far: float) -> bool:
        return time_so_far + self._prev_time < self.sec_per_op


def get_policy(name: str, params: Dict[str, Any]) -> Policy:
    name_lower = name.lower()
    if name_lower == 'no_prediction':
        return NoPredictionPolicy(ops_per_sec=params['ops_per_sec'], name=name)
    elif name-lower == 'previous_measurement':
        return PreviousMeasurementPolicy(ops_per_sec=params['ops_per_sec'], name=name)
    raise ValueError(f'Unknown policy name: {name}')


