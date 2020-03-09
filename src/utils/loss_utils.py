import tensorflow as tf

from .constants import SMALL_NUMBER

def binary_classification_loss(predicted_probs: tf.Tensor,
                               predictions: tf.Tensor, 
                               labels: tf.Tensor, 
                               pos_weight: float, 
                               neg_weight: float) -> tf.Tensor:
        """
        Computes a weighted binary cross entropy loss to bias towards specific types of errors.

        Args:
            predicted_probs: A [B, 1] tensor of predicted probabilities
            predictions: A [B, 1] tensor of predicted labels
            labels: A [B, 1] tensor of expected labels
            pos_weight: Weight to multiply on false positives
            neg_weight: Weight to multiply on false negatives
        """
        log_probs = -tf.log(predicted_probs)
        log_opp_probs = -tf.log(1.0 - predicted_probs)
        return tf.where((1.0 - tf.abs(predictions - labels)) < SMALL_NUMBER,
                        x=neg_weight * labels * log_probs + pos_weight * (1.0 - labels) * log_opp_probs,  # False negative or False Positive
                        y=labels * log_probs + (1.0 - labels) * log_opp_probs)


def f1_score_loss(predicted_probs: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """
    Computes a loss function based on F1 scores (harmonic mean of precision an recall).

    Args:
        predicted_probs: A [B, 1] tensor of predicted probabilities
        labels: A [B, 1] tensor of expected labels
    Returns:
        A tensor of sample-wise losses
    """
    # probability-based assessment of true positives, false negatives, and false positives
    # we use probabilities instead of thresholded labels to ensure that the loss function is differentiable
    true_positives = predicted_probs * labels
    false_negatives = (1.0 - predicted_probs) * labels
    false_positives = predicted_probs * (1.0 - labels)

    precision = true_positives / (true_positives + false_positives + SMALL_NUMBER)
    recall = true_positives / (true_positives + false_negatives + SMALL_NUMBER)

    f1_score = 2 * precision * recall / (precision + recall + SMALL_NUMBER)
    return 1.0 - f1_score
