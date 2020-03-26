import numpy as np

from utils.constants import SMALL_NUMBER


def threshold_sigmoid(predictions: np.ndarray, thresholds: np.ndarray, sharpen_factor: float) -> np.ndarray:
    exp_diff = np.exp(sharpen_factor * (predictions - thresholds))
    return exp_diff / (1.0 + exp_diff)


def array_product(array: np.ndarray, axis: int) -> np.ndarray:
    log_array = np.log(array)
    return np.exp(np.sum(log_array, axis=axis))


def f1_loss(predictions: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, sharpen_factor: float) -> float:
    """
    Returns a differentiable version of a F1-score which can be optimized using gradient-based methods.

    Args:
        predictions: A [N, L] matrix of predictions for each output level (as produced by the adaptive model)
        labels: A [N] array of the true labels (i.e. 0 / 1)
        thresholds: A [L] array of the current thresholds
        sharpen_factor: Factor by which to sharpen the sigmoid function
    """
    expanded_thresholds = np.expand_dims(thresholds, axis=0)  # [1, L]
    thresholded_predictions = threshold_sigmoid(predictions, expanded_thresholds, sharpen_factor)  # [N, L]
    model_predictions = array_product(thresholded_predictions, axis=-1)  # [N]

    label_prediction_sum = 2 * np.sum(labels * model_predictions)
    prediction_sum = np.sum(model_predictions)
    label_sum = np.sum(labels)

    return 1.0 - (label_prediction_sum / (prediction_sum + label_sum + SMALL_NUMBER))


def f1_loss_gradient(predictions: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, sharpen_factor: float) -> np.ndarray:
    """
    Computes the derivative of the F1 score loss function (in the method above).

    Args:
        predictions: A [N, L] matrix of predictions for each output level (as produced by the adaptive model)
        labels: A [N] array of the true labels (i.e. 0 / 1)
        thresholds: A [L] array of the current thresholds
        sharpen_factor: Factor by which to sharpen the sigmoid function
    Returns:
        An [L] array representing the threshold gradients.
    """
    # Get the model prediction using a sharpened sigmoid function
    expanded_thresholds = np.expand_dims(thresholds, axis=0)  # [1, L]
    thresholded_predictions = threshold_sigmoid(predictions, expanded_thresholds, sharpen_factor)  # [N, L]
    model_predictions = array_product(thresholded_predictions, axis=-1)  # [N]

    expanded_labels = np.expand_dims(labels, axis=-1)

    # Compute the threshold derivatives
    threshold_opp = 1.0 - thresholded_predictions  # [N, L]
    threshold_derivative = -1 * sharpen_factor * (threshold_opp) * np.expand_dims(model_predictions, axis=1)  # [N, L]

    # Compute building blocks of the derivative
    label_derivative_sum = np.sum(threshold_derivative * expanded_labels, axis=0)  # [L]
    derivative_sum = np.sum(threshold_derivative, axis=0)  # [L]
    label_prediction_sum = np.sum(model_predictions * labels)  # Scalar
    label_sum = np.sum(labels)  # Scalar
    prediction_sum = np.sum(model_predictions)  # Scalar

    # Compute derivative via the Quotient Rule
    numerator = 2 * label_derivative_sum * (label_sum + prediction_sum) - 2 * label_prediction_sum * derivative_sum  # [L]
    denominator = np.square(label_sum + prediction_sum)

    return -1 * numerator / (denominator + SMALL_NUMBER)
