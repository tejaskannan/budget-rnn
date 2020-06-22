import numpy as np
from typing import Union, List


def sample_sequence(sequence: Union[np.ndarray, List[np.ndarray]], seq_length: int) -> np.ndarray:
    """
    Samples the given sequence to the provided length. The chosen samples are in the same order
    as the original sequence.
    """
    if len(sequence) <= seq_length:
        return sequence

    indices = list(range(len(sequence)))
    selected_indices = np.sort(np.random.choice(indices, size=seq_length, replace=False))
    return np.array([sequence[i] for i in selected_indices])


def sample_sequence_batch(batch: np.ndarray, seq_length: int) -> np.ndarray:
    """
    Samples the batch of sequences so that each sequence is the given length. The chosen samples are in the same
    order as the original sequence.

    Args:
        batch: A [B, T, D] array containing feature vectors (D) for each sequence element (T) in the batch (B)
        seq_length: The desired length of the sequence (K).
    Returns:
        A [B, K, D] array that holds the sampled batch.
    """
    if batch.shape[1] <= seq_length:
        return batch

    indices = list(range(batch.shape[1]))
    selected_indices = [np.sort(np.random.choice(indices, size=seq_length, replace=False)) for _ in range(batch.shape[0])]

    sampled_batch: List[np.ndarray] = []
    for sequence, sampled_indices in zip(batch, selected_indices):
        sampled_batch.append(sequence[sampled_indices])

    return np.array(sampled_batch)  # [B, K, D]


def batch_sample_noise(input_features: np.ndarray, noise_weight: float) -> np.ndarray:
    """
    Applies noise by combining features in the batch using a weighted average. The idea
    is that a small fraction of features from a different sample should not change
    the classification label.

    Args:
        inputs_features: A [B, T, D] array of features vectors with dimension D for each sequence element (T) in the batch (B)
        noise_weight: A value in [0, 1) to use as the weighted average weight.
    Returns:
        A [B, T, D] array of noisy features.
    """
    assert noise_weight >= 0 and noise_weight < 1, 'Noise weight must be in [0, 1). Got {0}'.format(noise_weight)

    input_copy = np.copy(input_features)
    np.random.shuffle(input_copy)

    return (1.0 - noise_weight) * input_features + noise_weight * input_copy
