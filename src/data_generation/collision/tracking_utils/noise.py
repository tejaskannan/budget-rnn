import numpy as np


def salt_pepper_noise(image: np.array, p: float) -> np.array:
    """
    Applies salt-and-pepper noise to the given image.

    Args:
        image: Raw image. Represented as a 2D numpy array of pixels.
        p: Probability that a pixel will be set to white or black. Either
           color is chosen uniformly at random.
    Returns:
        The noisy image.
    """
    assert p >= 0.0 and p <= 1.0, 'The probability must in in [0, 1]'

    black_prob = p / 2
    white_prob = 1.0 - black_prob

    noisy_image = np.zeros_like(image)

    for i in range(noisy_image.shape[0]):
        for j in range(noisy_image.shape[1]):
            r = np.random.random()

            if r < black_prob:
                noisy_image[i][j] = 0
            elif r > white_prob:
                noisy_image[i][j] = 255
            else:
                noisy_image[i][j] = image[i][j]

    return noisy_image
