import numpy as np


class LinearUCB:

    def __init__(self, num_arms: int, num_features: int, alpha: float):
        self.alpha = alpha
        self.num_arms = num_arms  # K
        self.num_features = num_features  # D
        self.bs = [np.zeros(shape=(num_features,)) for _ in range(num_arms)]  # List of K, [D] arrays
        self.As = [np.eye(num_features) for _ in range(num_arms)]  # List of K, [D, D] matrices

    def predict(self, x: np.ndarray) -> int:
        scores: List[float] = []

        for arm in range(self.num_arms):
            A_inv = np.linalg.inv(self.As[arm])
            b = self.bs[arm]

            theta = A_inv.dot(b)
            p = theta.T.dot(x) + self.alpha * np.sqrt(x.T.dot(A_inv).dot(x))
            scores.append(p)

        return np.argmax(scores)

    def update(self, arm: int, reward: float, x: np.ndarray):
        self.As[arm] = self.As[arm] + x.T.dot(x)
        self.bs[arm] = self.bs[arm] + reward * x
