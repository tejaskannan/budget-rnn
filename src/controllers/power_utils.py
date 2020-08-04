import numpy as np

# Constants from taking readings from the DHT-11 sensor
VCC = 3.3                  # Supply Voltage (3.3 V)
FREQ = 0.5                 # Sample Freq (Hz)
SAMPLE_CURRENT = 0.8566    # Current when taking a sample
DEFAULT_CURRENT = 0.055    # Current when skipping a sample (based on LPM0)


def get_energy(num_samples: int, seq_length: int) -> float:
    current_on = num_samples * SAMPLE_CURRENT
    current_off = (seq_length - num_samples) * DEFAULT_CURRENT
    sample_time = 1.0 / FREQ
    return VCC * sample_time * (current_on + current_off)


def get_avg_power(num_samples: int, seq_length: int, multiplier: int = 1) -> float:
    assert num_samples > 0, 'Must have a positive number of samples'
    sample_time = 1.0 / FREQ
    return get_energy(num_samples * multiplier, seq_length) / (sample_time * seq_length)


def get_avg_power_multiple(num_samples: np.ndarray, seq_length: int, multiplier: int = 1) -> float:
    """
    Computes the weighted average power over the number of samples per episode.

    Args:
        num_samples: An [B] array of the number of samples for each element
        seq_length: The total sequence length
        multiplier: A factor to multiply the number of sample by. This argument is helpful when
            then number of samples refer to a group of samples (i.e. subsample fraction).
    Returns:
        The weighted average power.
    """
    assert multiplier >= 1, 'Multiplier must be >= 1'
    
    min_num_samples = np.min(num_samples)
    assert min_num_samples > 0, 'Cannot have zero samples'

    max_num_samples = np.max(num_samples) * multiplier
    assert max_num_samples <= seq_length, 'Can have at most {0} samples'.format(seq_length)

    sample_counts = np.bincount(num_samples * multiplier, minlength=seq_length + 1)  # [T]
    sample_weights = sample_counts.astype(float) / np.sum(sample_counts)

    avg_power = 0.0
    for idx in range(1, seq_length + 1):
        sample_power = get_avg_power(idx, seq_length)
        avg_power += sample_weights[idx] * sample_power

    return avg_power
