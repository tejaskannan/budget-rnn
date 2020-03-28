import numpy as np

from utils.constants import SMALL_NUMBER, BIG_NUMBER


def threshold_sigmoid(predictions: np.ndarray, thresholds: np.ndarray, sharpen_factor: float) -> np.ndarray:
    exp_diff = np.exp(sharpen_factor * (predictions - thresholds))
    return exp_diff / (1.0 + exp_diff)


def array_product(array: np.ndarray, axis: int) -> np.ndarray:
    log_array = np.log(array)
    return np.exp(np.sum(log_array, axis=axis))


def computed_levels(predictions: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    diff = predictions - np.expand_dims(thresholds, axis=0)  # [N, L]
    diff_mask = (diff >= 0.0).astype(np.float32) * BIG_NUMBER  # [N, L]
    indices = np.expand_dims(np.arange(start=0, stop=len(thresholds)), axis=0)  # [1, L]

    # Apply mask and return minimum
    masked_indices = indices + diff_mask
    levels = np.min(masked_indices, axis=-1)  # [N]
    return np.minimum(levels, len(thresholds))  # [N]


def level_penalty(predictions: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    levels = computed_levels(predictions, thresholds)
    return levels / (len(thresholds) - 1)


def level_penalty_gradient(predictions: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    diff = predictions - np.expand_dims(thresholds, axis=0)  # [N, L]
    levels = np.expand_dims(computed_levels(predictions, thresholds), axis=-1)  # [N, 1]
    num_thresholds = len(thresholds)

    indices = np.expand_dims(np.arange(start=0, stop=num_thresholds), axis=0)  # [1, L]
    indices = np.tile(indices, reps=(len(predictions), 1))  # [N, L]

    # Apply weight increase to all indices less than the chosen index
    increase_mask = (indices < levels).astype(np.float32)  # [N, L]
    increase_weight = (levels - indices) / num_thresholds  # [N, L]
    increase_grad = increase_mask * increase_weight * diff  # [N, L]

    return increase_grad


def f1_loss(predictions: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, sharpen_factor: float, level_weight: float) -> float:
    """
    Returns a differentiable version of a F1-score which can be optimized using gradient-based methods.

    Args:
        predictions: A [N, L] matrix of predictions for each output level (as produced by the adaptive model)
        labels: A [N] array of the true labels (i.e. 0 / 1)
        thresholds: A [L] array of the current thresholds
        sharpen_factor: Factor by which to sharpen the sigmoid function
        level_weight: Weight to place on argmin loss for zero classifications
    Returns:
        The f1-based loss function
    """
    expanded_thresholds = np.expand_dims(thresholds, axis=0)  # [1, L]
    thresholded_predictions = threshold_sigmoid(predictions, expanded_thresholds, sharpen_factor)  # [N, L]
    model_predictions = array_product(thresholded_predictions, axis=-1)  # [N]

    label_prediction_sum = 2 * np.sum(labels * model_predictions)
    prediction_sum = np.sum(model_predictions)
    label_sum = np.sum(labels)

    zero_labels = 1.0 - labels  # [N]
    level_sample_loss = level_penalty(predictions, thresholds) * zero_labels  # [N]
    level_loss = level_weight * (np.sum(level_sample_loss) / (np.sum(zero_labels) + SMALL_NUMBER))

    return 1.0 - (label_prediction_sum / (prediction_sum + label_sum + SMALL_NUMBER)) + level_loss


def f1_loss_gradient(predictions: np.ndarray, labels: np.ndarray, thresholds: np.ndarray, sharpen_factor: float, level_weight: float) -> np.ndarray:
    """
    Computes the derivative of the F1 score loss function (in the method above).

    Args:
        predictions: A [N, L] matrix of predictions for each output level (as produced by the adaptive model)
        labels: A [N] array of the true labels (i.e. 0 / 1)
        thresholds: A [L] array of the current thresholds
        sharpen_factor: Factor by which to sharpen the sigmoid function
        level_weight: Weight to place on level loss for zero classifications
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
    zero_labels = np.expand_dims(1.0 - labels, axis=-1)  # [N, 1]
    level_sample_grad = level_penalty_gradient(predictions, thresholds) * zero_labels  # [N, L]
    level_grad = level_weight * np.sum(level_sample_grad, axis=0) / np.sum(zero_labels)  # [L]

    return -1 * numerator / (denominator + SMALL_NUMBER) - level_grad
