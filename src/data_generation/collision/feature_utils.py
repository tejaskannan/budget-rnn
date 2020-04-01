import numpy as np
import cv2
from typing import List


def average_pool(array: np.ndarray, num_chunks: int) -> List[float]:
    features: List[float] = []

    x_stride = int(array.shape[0] / num_chunks)
    y_stride = int(array.shape[1] / num_chunks)

    for i in range(num_chunks):
        for j in range(num_chunks):
            x = i * x_stride
            y = j * y_stride

            chunk = array[x:x+x_stride, y:y+y_stride]
            avg = np.average(chunk)

            features.append(avg)

    return features


def color_histogram(img: np.ndarray, size: int) -> np.ndarray:
    features: List[np.ndarray] = []

    for i in range(img.shape[-1]):
        channel = img[:, :, i]
        hist = cv2.calcHist([channel], channels=[0], mask=None, histSize=[size], ranges=[0, 256])
        features.append(hist)

    # Return a 1-dimensional array
    return np.vstack(features).reshape(-1)


def apply_gabor_filter(img: np.ndarray, filter_size: int, scale: float, angle: float) -> np.ndarray:
    kernel = cv2.getGaborKernel((filter_size, filter_size), scale, angle, 1.0, 0.5, 0, ktype=cv2.CV_32F)
    return cv2.filter2D(img, cv2.CV_8UC3, kernel)


def apply_laplace_filter(img: np.ndarray):
    return cv2.Laplacian(img, cv2.CV_32F)
