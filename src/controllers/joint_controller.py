import numpy as np

from dataset.dataset import Dataset, DataSeries
from models.adaptive_model import AdaptiveModel
from threshold_optimization.optimize_thresholds import get_serialized_info
from logistic_regression_controller import Controller, POWER, fetch_model_states
from utils.constants import BIG_NUMBER, SMALL_NUMBER, NUM_CLASSES


### Constants ###
BETA_POS = 200
BETA_NEG = 0.1
ONE_HALF = 0.5

### Utility functions ###

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted_x = x - np.max(x, axis=-1, keepdims=True)  # Shift numbers for numerical stability
    x_exp = np.exp(shifted_x)
    return x_exp / np.sum(x_exp, axis=-1, keepdims=True)


def softmax_derivative(x: np.ndarray) -> np.ndarray:
    """
    Computes the softmax derivative along the last axis.

    Args:
        x: A [B, N] array
    """
    n = x.shape[-1]
    softmax_x = softmax(x, axis=-1)  # [B, N]

    expanded_last = np.expand_dims(softmax_x, axis=-1)  # [B, N, 1]
    expanded_first = np.expand_dims(softmax_x, axis=-2)  # [B, 1, N]
    identity = np.expand_dims(np.eye(n), axis=0)  # [1, N, N]

    jacobian = expanded_last * (identity - expanded_first)
    return jacobian


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))        


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1.0 - sigmoid_x)


### Controller class ###

class JointController:

    def __init__(self,
                 model_path: str,
                 dataset_folder: str,
                 precision: bool,
                 budget: float,
                 max_iter: int,
                 regularization_weight: float,
                 power_violation_weight: float,
                 step_size: float,
                 momentum: float,
                 anneal_rate: float):
        self._model_path = model_path
        self._dataset_folder = dataset_folder

        # Load the model and dataset
        model, dataset, _ = get_serialized_info(model_path, dataset_folder=dataset_folder)

        self._model = model
        self._dataset = dataset
        self._is_fitted = False
        self._num_levels = model.num_outputs
        self._num_features = model.hypers.model_params['state_size']
        self._rand = np.random.RandomState(seed=42)
        self._regularization_weight = regularization_weight
        self._power_violation_weight = power_violation_weight
        self._step_size = step_size
        self._momentum = momentum
        self._anneal_rate = anneal_rate

        self._budget = budget
        self._precision = precision
        self._max_iter = max_iter
        self._W = None  # Trainable model parameters

    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        """
        Computes the predicted probability for this sample.
        Args:
            X: A [B, L, D] array of states for each level
        Returns:
            The predicted probability for each level. This is a [L] array.
        """
        assert self._W is not None, 'Model is not fitted.'

        # If needed we append the bias term
        if X.shape[-1] < self._W.shape[-1]:
            bias_shape = X.shape[:-1] + (1, )
            ones = np.ones(shape=bias_shape)
            X = np.concatenate([X, ones], axis=-1)

        # We compute the predictions using matrix operations. This leverages numpy optimizations.
        X_transpose = np.transpose(X, axes=[0, 2, 1])
        linear_combination = np.matmul(self._W, X_transpose)  # [B, L, L]
        linear_vector = np.diagonal(linear_combination, axis1=1, axis2=2)  # [B, L]

        # Apply the logistic function
        return sigmoid(linear_vector)

    def accuracy(self, predicted_probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the accuracy of the given probabilities.

        Args:
            predicted_probs: A [B, L] array of predicted probabilities for each level
            y: A [B, L] array of labels for each level. The labels are whether the model got the answer
                correct at each level.
        """
        # 0/1 array indicating the level to choose (as selected by the model)
        level_predictions = (predicted_probs >= ONE_HALF).astype(float)  # [B, L]

        # Select the levels for prediction
        level_indices = np.expand_dims(np.arange(predicted_probs.shape[-1]), axis=0)  # [1, L]
        masked_indices = ((1.0 - level_predictions) * BIG_NUMBER) + level_indices  # [B, L]
        chosen_levels = np.min(masked_indices, axis=-1).astype(int)  # [B]
        chosen_levels = np.clip(chosen_levels, a_min=0, a_max=y.shape[-1] - 1)  # [B]

        # Select the predictions corresponding to the chosen levels and compute the accuracy
        batch_indices = np.arange(chosen_levels.shape[0])
        selected_predictions = y[batch_indices, chosen_levels]  # [B]
        accuracy = np.average(selected_predictions)

        return accuracy

    def loss_function(self, X: np.ndarray, y: np.ndarray, predicted_probs: np.ndarray) -> float:
        """
        Computes the loss function on the given inputs and outputs

        Args:
            X: A [B, L, D] array of states for each level
            y: A [B, L] array of labels for each level
            predicted_probs: [B, L] array of predicted probabilities for each level. This is the result
                of the predict_prob function. We pass this value in to avoid re-computation.
        Returns:
            A [B] array containing the losses for each sample.
        """
        assert self._W is not None, 'Model is not fitted.'

        # We want to turn off the loss for false negatives. This incentivizes the model to achieve 100% accuracy. The power restrictions
        # should limit how aggressive the settings really are. Furthermore, we heavily penalize false positives, as these block
        # the system from accessing higher levels (and are thus costly).
        shifted_y = (2 * y) - 1
        level_predictions = (predicted_probs >= ONE_HALF).astype(float)
        
        false_neg_mask = np.where(np.logical_and(np.isclose(level_predictions, 0), np.isclose(y, 1)), BETA_NEG, 1)
        false_pos_mask = np.where(np.logical_and(np.isclose(level_predictions, 1), np.isclose(y, 0)), BETA_POS, 1)
        label_mask = shifted_y * false_neg_mask * false_pos_mask * 0.5

        level_loss = -1 * (predicted_probs - ONE_HALF) * label_mask

        logistic_loss = -1 * np.sum(level_loss, axis=-1)  # [B]
        logistic_loss = np.average(logistic_loss)

        # Compute the regularization term (scalar)
        regularization_loss = ONE_HALF * self._regularization_weight * np.square(np.linalg.norm(self._W, ord='fro'))

        

        # Compute the level-wise prediction terms
       # level_indices = np.expand_dims(self._num_levels - np.arange(start=1, stop=self._num_levels + 1), axis=0)  # [1, L]
       # mask = -BIG_NUMBER * (predicted_probs < ONE_HALF)  # [B, L]
       # level_weights = softmax((mask + predicted_probs) * BETA * level_indices, axis=-1)  # [B, L]

       # # Compute the power loss term
       # level_power_approx = np.sum(np.expand_dims(POWER, axis=0) * level_weights, axis=-1)  # [B]
       # power_approx = np.average(level_power_approx)  # Scalar
       # power_loss = np.clip(self._power_violation_weight * (power_approx - self._budget), a_min=0, a_max=None)

        return logistic_loss + regularization_loss

    def loss_gradient(self, X: np.ndarray, y: np.ndarray, predicted_probs: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the loss function on the given inputs and outputs.

        Args:
            X: A [B, L, D] array of states for each level. B is the batch size
            y: A [B, L] array of labels for each level
            predicted_probs: A [B, L] array of predictions for the current weights. This is the result of the
                predict_prob function. We pass this to avoid re-computation.
        Returns:
            A [L, D] array containing the derivative with respect to the model parameters.
        """
        # Compute derivative for the logistic term
        # predicted_diff = np.expand_dims(predicted_probs - y, axis=-1)  # [B, L, 1]
        # logistic_sample_grad = X * predicted_diff  # [B, L, D]
        # logistic_grad = np.average(logistic_sample_grad, axis=0)  # [L, D]

        level_predictions = (predicted_probs >= ONE_HALF).astype(float)

        shifted_y = (2 * y) - 1
        false_neg_mask = np.where(np.logical_and(np.isclose(level_predictions, 0), np.isclose(y, 1)), BETA_NEG, 1)
        false_pos_mask = np.where(np.logical_and(np.isclose(level_predictions, 1), np.isclose(y, 0)), BETA_POS, 1)
        label_mask = shifted_y * false_neg_mask * false_pos_mask * 0.5

        sigmoid_grad = predicted_probs * (1.0 - predicted_probs)  # [B, L]
        weighted_sigmoid_grad = np.expand_dims(-1 * sigmoid_grad * label_mask, axis=-1)  # [B, L, 1]

        logistic_sample_grad = X * weighted_sigmoid_grad  # [B, L, D]
        logistic_grad = np.average(logistic_sample_grad, axis=0)  # [L, D]

        # Compute derivative for the regularization term
        regularization_grad = self._regularization_weight * self._W  # [L, D]

        # Compute derivative of the power penalty term. We set the derivative to all zeros if
        # the approximate power is under budget.
       # level_indices = self._num_levels - np.arange(start=1, stop=self._num_levels + 1)  # [L]
       # mask = -BIG_NUMBER * (predicted_probs < ONE_HALF)  # [L]
       # shifted_probs = (mask + predicted_probs) * BETA * level_indices
       # power_approx = np.sum(POWER * softmax(shifted_probs))  # Scalar

       # if power_approx > self._budget:
       #     power_softmax_derivative = softmax_derivative(shifted_probs)  # [L, L]

       #     power_weight_derivative = (BETA * level_indices) * sigmoid_derivative(predicted_probs)  # [L]
       #     power_weight_derivative = np.expand_dims(power_weight_derivative * POWER, axis=1)  # [L, 1]
       #     power_derivative = self._power_violation_weight * power_softmax_derivative.dot(X * power_weight_derivative)  # [L, D]
       # else:
       #     power_derivative = np.zeros_like(self._W)

        return logistic_grad + regularization_grad

    def fit(self, series: DataSeries, patience: int):
        X_train, y_train, model_predictions = fetch_model_states(self._model, self._dataset, series=series)

        # Add the bias term
        if X_train.shape[-1] < self._num_features + 1:
            bias_shape = X_train.shape[:-1] + (1, )
            ones = np.ones(shape=bias_shape)
            X_train = np.concatenate([X_train, ones], axis=-1)

        # Initialize the [L, D] weight matrix
        self._W = self._rand.uniform(low=-0.7, high=0.7, size=(self._num_levels, self._num_features + 1))
        step_size = self._step_size
        # avg_gradient = np.zeros_like(self._W)

        for i in range(self._max_iter):
            prev_W = np.copy(self._W)

            # Results using the current parameters
            predicted_probs = self.predict_prob(X_train)

            # Calculate the loss and gradient over the entire data
            loss = self.loss_function(X_train, y_train, predicted_probs)  # Scalar
            grad = self.loss_gradient(X_train, y_train, predicted_probs)  # [L, D]
            acc = self.accuracy(predicted_probs, y_train)  # [L]

            # Perform the gradient descent step
            # avg_gradient = self._momentum * avg_gradient + (1.0 - self._momentum) * np.square(grad)
            # adaptive_step_size = step_size / np.sqrt(avg_gradient + SMALL_NUMBER)
            self._W = self._W - step_size * grad
            step_size = step_size * self._anneal_rate

            print('Iteration {0}: Loss -> {1:.5f}, Accuracy -> {2:.5f}'.format(i + 1, loss, np.average(acc)), end='\r')

            if np.isclose(self._W, prev_W).all():
                print()
                print('Converged.')
                break

        print()
        # print(predicted_probs)


# Run some tests
controller = JointController(model_path='../saved_models/fordA/29_06_2020/model-BIDIR_SAMPLE-FordA-2020-06-29-00-43-47_model_best.pkl.gz',
                             dataset_folder='../datasets/FordA/folds_spectrum',
                             precision=10,
                             budget=47,
                             max_iter=1000,
                             regularization_weight=0.0,
                             power_violation_weight=0.01,
                             step_size=0.1,
                             momentum=0.9,
                             anneal_rate=0.9)

controller.fit(series=DataSeries.VALID, patience=10)
