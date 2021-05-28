import numpy as np
from scipy.signal import hamming
from typing import List


def amplify_signal(signal: np.ndarray, emphasis: float):
    return np.append(signal[0], signal[1:] - emphasis * signal[:-1])


def frame_signal(signal: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    frames: List[np.ndarray] = []
    for i in range(0, len(signal), stride):
        window = signal[i:i+window_size]

        if len(window) == window_size:
            frames.append(window * hamming(M=window_size))

    return np.array(frames)


def sift(frames: np.ndarray) -> np.ndarray:
    nfft = frames.shape[1]
    transformed = np.abs(np.fft.rfft(a=frames, n=nfft, axis=-1))

    power_frames = (1.0 / nfft) * np.square(transformed)
    return power_frames



