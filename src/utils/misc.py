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
