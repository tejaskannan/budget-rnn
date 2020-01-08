from typing import Dict, Any


class Policy:

    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def should_continue(self, prediction_level: int, time_so_far: float) -> bool:
        raise NotImplementedError()


class OpsPerSecPolicy(Policy):

    def __init__(self, ops_per_sec: float, name: str):
        super().__init__(name)
        self._ops_per_sec = ops_per_sec

    @property
    def ops_per_sec(self) -> float:
        return self._ops_per_sec

    def should_continue(self, prediction_level: int, time_so_far: float) -> bool:
        # Ideally, we want to estimate how long the next operation will take and then make a decision
        return time_so_far < 1.0 / self._ops_per_sec 


def get_policy(name: str, params: Dict[str, Any]) -> Policy:
    name_lower = name.lower()
    if name_lower == 'ops_per_sec':
        return OpsPerSecPolicy(ops_per_sec=params['ops_per_sec'], name=name)
    raise ValueError(f'Unknown policy name: {name}')


