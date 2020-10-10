import numpy as np
from enum import Enum, auto
from typing import List

# Constants from taking readings from the DHT-11 sensor
TEMP_VCC = 3.3                  # Supply Voltage (3.3 V)
TEMP_FREQ = 0.5                 # Sample Freq (Hz)
TEMP_COLLECT_CURRENT = 0.8566    # Current when taking a sample
TEMP_BASE_CURRENT = 0.055    # Current when skipping a sample (based on LPM0)

# Constants based on bluetooth-based data collection
BT_ENERGY_PER_SAMPLE = 29.63  # Energy to collect a sample (mJ)
BT_BASE_POWER = 1.594  # Baseline power when system is idle (mW)
BT_FREQ = 0.5


class PowerType(Enum):
    TEMP = auto()
    BLUETOOTH = auto()


class PowerSystem:

    def __init__(self, total_levels: int, seq_length: int):
        self._total_levels = total_levels
        self._seq_length = seq_length
        self._multiplier = int(seq_length / total_levels)

    @property
    def total_levels(self) -> int:
        return self._total_levels

    @property
    def system_type(self) -> PowerType:
        raise NotImplementedError()

    @property
    def sample_period(self) -> float:
        raise NotImplementedError()

    def get_energy(self, num_samples: int) -> float:
        raise NotImplementedError()

    def get_max_power(self) -> float:
        return self.get_avg_power(self._total_levels)

    def get_min_power(self) -> float:
        return self.get_avg_power(1)

    def get_avg_power(self, num_levels: int) -> float:
        assert num_levels > 0, 'Must have a positive number of levels'

        total_time = self.sample_period * self._seq_length
        num_samples = num_levels * self._multiplier

        return self.get_energy(num_samples=num_samples) / total_time

    def get_avg_power_multiple(self, num_levels: np.ndarray) -> float:
        """
        Computes the weighted average power over the
        number of samples per episode.

        Args:
            num_samples: An [B] array of the number of
                levels for each batch element
        Returns:
            The weighted average power.
        """
        min_num_levels = np.min(num_levels)
        assert min_num_levels > 0, 'Cannot have zero samples'

        max_num_samples = np.max(num_levels) * self._multiplier
        assert max_num_samples <= self._seq_length, 'Can have at most {0} samples'.format(self._seq_length)

        counts = np.bincount(num_levels, minlength=self._total_levels + 1)  # [L + 1]
        weights = counts.astype(float) / np.sum(counts)  # [L + 1]

        return self.get_weighted_avg_power(weights[1:])

    def get_weighted_avg_power(self, weights: np.ndarray) -> float:
        """
        Computes the average power using a weighted average from the given weights.

        Args:
            sample_weights: A [L] array of weights for each level (L) (must form a distribution)
        Returns:
            The weighted average power
        """
        assert len(weights.shape) == 1 and weights.shape[0] == self._total_levels, 'Invalid shape: {0}'.format(weights.shape)

        estimates = self.get_power_estimates()
        weighted_avg = np.sum(weights * estimates)

        return weighted_avg

    def get_power_estimates(self) -> np.ndarray:
        estimates: List[float] = []

        for level_idx in range(1, self._total_levels + 1):
            power = self.get_avg_power(num_levels=level_idx)
            estimates.append(power)

        return np.array(estimates)


class BluetoothPowerSystem(PowerSystem):

    @property
    def system_type(self) -> PowerType:
        return PowerType.BLUETOOTH

    @property
    def sample_period(self) -> float:
        return 1.0 / BT_FREQ

    def get_energy(self, num_samples: int) -> float:
        total_time = self._seq_length * self.sample_period

        base_energy = BT_BASE_POWER * total_time
        collect_energy = BT_ENERGY_PER_SAMPLE * num_samples

        return collect_energy + base_energy


class TemperaturePowerSystem(PowerSystem):

    @property
    def system_type(self) -> PowerType:
        return PowerType.TEMP

    @property
    def sample_period(self) -> float:
        return 1.0 / TEMP_FREQ

    def get_energy(self, num_samples: int) -> float:
        current_on = num_samples * TEMP_COLLECT_CURRENT
        current_off = (self._seq_length - num_samples) * TEMP_BASE_CURRENT
        return TEMP_VCC * self.sample_period * (current_on + current_off)


def make_power_system(mode: PowerType, num_levels: int, seq_length: int) -> PowerSystem:
    if mode == PowerType.BLUETOOTH:
        return BluetoothPowerSystem(total_levels=num_levels, seq_length=seq_length)
    elif mode == PowerType.TEMP:
        return TemperaturePowerSystem(total_levels=num_levels, seq_length=seq_length)
    else:
        raise ValueError('Unknown power system type: {0}'.format(mode.name))
