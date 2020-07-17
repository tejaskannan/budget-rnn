import tensorflow as tf
import numpy as np


class DistributionPrior:

    def __init__(self, power: np.ndarray, target: float, learning_rate: float = 0.001):
        self._power = power
        self._input_units = len(power)
        self._target = target
        self._learning_rate = learning_rate
        self._sess = tf.Session(graph=tf.Graph())

        self._optimizer = None
        self._weights = None
        self._is_fitted = False
        self._is_made = False

    def make(self):
        with self._sess.graph.as_default():
            self._inputs = tf.get_variable(initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0, seed=52),
                                           shape=(self._input_units, ),
                                           name='input',
                                           trainable=True)
            self._normalized_inputs = tf.nn.softmax(self._inputs)  # [L]

            weighted_power = tf.reduce_sum(self._normalized_inputs * self._power)  # Scalar
            self._loss = tf.square(weighted_power - self._target)

            self._optimizer = tf.train.RMSPropOptimizer(learning_rate=self._learning_rate)
            self._training_step = self._optimizer.minimize(self._loss)
            self._is_made = True

    def init(self):
        with self._sess.graph.as_default():
            self._sess.run(tf.global_variables_initializer())

    def fit(self, n_iter: int = 5000, tol: float = 1e-5) -> np.ndarray:
        assert self._is_made, 'Must call make() first'

        with self._sess.graph.as_default():
            
            prev_weights = np.zeros(shape=(self._input_units, ))
            for i in range(n_iter):
                weights, _ = self._sess.run([self._normalized_inputs, self._training_step])

                if (np.abs(weights - prev_weights) < tol).all():
                    break

                prev_weights = np.copy(weights)

            self._weights = self._sess.run(self._normalized_inputs)
            self._is_fitted = True
            return self._weights

    def get_weights(self):
        assert self._is_fitted, 'Must call fit() first'
        return self._weights


if __name__ == '__main__':
    power = np.array([1, 2, 3])
    prior = DistributionPrior(power=power, target=1.64)
    prior.make()
    prior.init()
    w = prior.fit(n_iter=10000)

    print(w)
    print(np.sum(w * power))
