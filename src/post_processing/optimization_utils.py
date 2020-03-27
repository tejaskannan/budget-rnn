import numpy as np

from utils.constants import SMALL_NUMBER


def threshold_sigmoid(predictions: np.ndarray, thresholds: np.ndarray, sharpen_factor: float) -> np.ndarray:
    exp_diff = np.exp(sharpen_factor * (predictions - thresholds))
    return exp_diff / (1.0 + exp_diff)


def array_product(array: np.ndarray, axis: int) -> np.ndarray:
    log_array = np.log(array)
    return np.exp(np.sum(log_array, axis=axis))


def f1_loss(predictions: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, sharpen_factor: float, beta: float, argmin_weight: float) -> float:
    """
    Returns a differentiable version of a F1-score which can be optimized using gradient-based methods.

    Args:
        predictions: A [N, L] matrix of predictions for each output level (as produced by the adaptive model)
        labels: A [N] array of the true labels (i.e. 0 / 1)
        thresholds: A [L] array of the current thresholds
        sharpen_factor: Factor by which to sharpen the sigmoid function
        argmin_weight: Weight to place on argmin loss for zero classifications
        beta: Factor used during argmin calculation
    """
    expanded_thresholds = np.expand_dims(thresholds, axis=0)  # [1, L]
    thresholded_predictions = threshold_sigmoid(predictions, expanded_thresholds, sharpen_factor)  # [N, L]
    model_predictions = array_product(thresholded_predictions, axis=-1)  # [N]

    label_prediction_sum = 2 * np.sum(labels * model_predictions)
    prediction_sum = np.sum(model_predictions)
    label_sum = np.sum(labels)

    masked_argmin = soft_argmin(predictions, thresholds, beta=beta) * (1.0 - labels)  # [N]
    argmin_loss = argmin_weight * np.sum(masked_argmin, axis=0) / len(thresholds)  # Scalar

    return 1.0 - (label_prediction_sum / (prediction_sum + label_sum + SMALL_NUMBER)) + argmin_loss


def f1_loss_gradient(predictions: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, sharpen_factor: float, beta: float, argmin_weight: float) -> np.ndarray:
    """
    Computes the derivative of the F1 score loss function (in the method above).

    Args:
        predictions: A [N, L] matrix of predictions for each output level (as produced by the adaptive model)
        labels: A [N] array of the true labels (i.e. 0 / 1)
        thresholds: A [L] array of the current thresholds
        sharpen_factor: Factor by which to sharpen the sigmoid function
        beta: Factor used during argmin computation
        argmin_weight: weight to place on argmin loss for zero classifications
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

    # Compute gradient for the argmin penalty
    sample_argmin_grad = soft_argmin_gradient(predictions, thresholds, beta=beta) * (1.0 - expanded_labels)  # [N, L]
    argmin_grad = argmin_weight * np.sum(sample_argmin_grad, axis=0)  # [L]

    return -1 * numerator / (denominator + SMALL_NUMBER) + argmin_grad


def soft_argmin(predictions: np.ndarray, thresholds: np.ndarray, beta: float) -> np.ndarray:
    """
    Differentiable approximation of an argmin operation

    Args:
        predictions: A [N, L] array of model predictions
        thresholds: A [L] array of output thresholds
        beta: Factor that controls level of approximation (large values are better approximates but have lower numerical stability)
    Returns:
        A [N] array containing the approximate argmin for each entry
    """
    diff = predictions - np.expand_dims(thresholds, axis=0)  # [N, L]
    exp_diff = np.exp(beta * diff)  # [N, L]

    exp_sum = np.sum(exp_diff, axis=-1, keepdims=True)  # [N, 1]

    num_thresholds = len(thresholds)
    indices = num_thresholds - np.expand_dims(np.arange(start=1, stop=num_thresholds + 1), axis=0)  # [1, L]
    approx_argmin = np.sum(indices * exp_diff / (exp_sum + SMALL_NUMBER), axis=-1)  # [L]

    return approx_argmin


def soft_argmin_gradient(predictions: np.ndarray, thresholds: np.ndarray, beta: float) -> np.ndarray:
    """
    Returns the gradient of the soft argmin function (above) with respect to the given thresholds.

    Args:
        predictions: A [N, L] matrix of model predictions
        thresholds: A [L] array of output thresholds
        beta: Factor that controls the level of approximation
    Returns:
        A [N, L] array containing the gradient of thresholds for each sample
    """
    diff = predictions - np.expand_dims(thresholds, axis=0)  # [N, L]
    exp_diff = np.exp(beta * diff)  # [N, L]

    exp_sum = np.sum(exp_diff, axis=-1, keepdims=True)  # [N, 1]
    exp_max = exp_diff / (exp_sum + SMALL_NUMBER)  # [N, L]

    num_thresholds = len(thresholds)
    indices = num_thresholds - np.expand_dims(np.arange(start=1, stop=num_thresholds + 1), axis=0)  # [1, L]

    return (indices) * (-beta) * exp_max * (1.0 - exp_max)  # [N, L]
