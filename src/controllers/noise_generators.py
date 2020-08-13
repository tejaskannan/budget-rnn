import numpy as np
from typing import Optional, Dict, Any


class NoiseGenerator:

    def __init__(self, max_time: int, loc: float, scale: float, seed: int):
        self._max_time = max_time
        self._loc = loc
        self._scale = scale

        self._rand = np.random.RandomState(seed=seed)
        self._gaussian_noise = self._rand.normal(loc=loc, scale=scale, size=(max_time, ))

    def get_noise(self, t: int) -> float:
        raise NotImplementedError()

    def __str__(self) -> str:
        return type(self).__name__


class GaussianNoise(NoiseGenerator):

    def get_noise(self, t: int) -> float:
        return self._gaussian_noise[t]

    def __str__(self) -> str:
        return 'Gaussian: Loc -> {0:.4f}, Scale -> {1:.4f}'.format(self._loc, self._scale)


class SinusoidalNoise(NoiseGenerator):

    def __init__(self, period: int, amplitude: float, max_time: int, loc: float, scale: float, seed: int):
        super().__init__(max_time, loc, scale, seed)
        self._period = (2 * np.pi) / period
        self._amplitude = amplitude

    def get_noise(self, t: int) -> float:
        sin_noise = self._amplitude * np.sin(self._period * t)
        return sin_noise + self._gaussian_noise[t]

    def __str__(self) -> str:
        return 'Sinusoidal: Loc -> {0:.4f}, Scale -> {1:.4f}, Period -> {2:.4f}, Amp -> {3:.4f}'.format(self._loc, self._scale, self._period, self._amplitude)


class SquareNoise(NoiseGenerator):

    def __init__(self, period: int, amplitude: float, max_time: int, loc: float, scale: float, seed: int):
        super().__init__(max_time, loc, scale, seed)
        self._period = period
        self._amplitude = amplitude

    def get_noise(self, t: int) -> float:
        parity = int(t / self._period) % 2
        sign = 2 * parity - 1
        return sign * self._amplitude + self._gaussian_noise[t]

    def __str__(self) -> str:
        return 'Square: Loc -> {0:.4f}, Scale -> {1:.4f}, Period -> {2:.4f}, Amp -> {3:.4f}'.format(self._loc, self._scale, self._period, self._amplitude)


def get_noise_generator(noise_params: Dict[str, Any], max_time: int, seed: int = 48) -> NoiseGenerator:
    # Unpack arguments
    loc = noise_params['loc']
    scale = noise_params['scale']
    period = noise_params.get('period')
    amplitude = noise_params.get('amplitude')

    name = noise_params['noise_type'].lower()
    if name == 'gaussian':
        return GaussianNoise(max_time=max_time, loc=loc, scale=scale, seed=seed)
    elif name == 'sin':
        assert period is not None, 'Must provide a period for sin noise.'
        assert amplitude is not None, 'Must provide an amplitude for sin noise.'
        return SinusoidalNoise(period=period,
                               amplitude=amplitude,
                               max_time=max_time,
                               loc=loc,
                               scale=scale,
                               seed=seed)
    elif name == 'square':
        assert period is not None, 'Must provide a period for square noise.'
        assert amplitude is not None, 'Must provide an amplitude for square noise.'
        return SquareNoise(period=period,
                           amplitude=amplitude,
                           max_time=max_time,
                           loc=loc,
                           scale=scale,
                           seed=seed)
    else:
        raise ValueError('Unknown noise type: {0}'.format(name))
