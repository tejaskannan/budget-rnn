import numpy as np
from scipy import optimize
from utils.constants import SMALL_NUMBER

MIN_LEARNING_RATE = 0.01

def _softmax(x: np.ndarray) -> np.ndarray:
    x_max = np.max(x)
    x_exp = np.exp(x - x_max)
    return x_exp / np.sum(x_exp)


class PowerDistribution:

    def __init__(self, power: np.ndarray, target: float):
        self._power = np.reshape(power, (-1, 1))
        self._units = self._power.shape[0]
        self._target = target

        self._rand = np.random.RandomState(seed=37)
        self._weights = None
        self._is_fitted = False
    
    def fit(self, n_iter: int = 1000, tol: float = 1e-5) -> np.ndarray:
        # Initialize the weights
        weights = self._rand.uniform(low=0.1, high=1.0, size=(self._units, ))
        weights = weights / np.sum(weights)

        ones = np.ones_like(weights).T  # [1, L]

        bounds = optimize.Bounds(lb=0, ub=np.inf)
        simplex_constraint = optimize.LinearConstraint(A=ones, lb=1, ub=1)

        result = optimize.minimize(fun=lambda x: np.square(self._power.T.dot(x) - self._target),
                                   x0=weights,
                                   method='SLSQP',
                                   bounds=bounds,
                                   constraints=simplex_constraint)
        self._weights = result.x
        return self._weights

    def get_weights(self):
        assert self._is_fitted, 'Must call fit() first'
        return self._weights


if __name__ == '__main__':
    power = np.array([0.446028, 0.710556, 0.975084, 1.239612, 1.50414, 1.768668, 2.033196, 2.297724, 2.562252, 2.82678])
    prior = PowerDistribution(power=power, target=1.5)
    w = prior.fit()

    print(w)
    print(np.sum(w * power))
