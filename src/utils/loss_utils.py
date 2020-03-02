import tensorflow as tf

from .constants import SMALL_NUMBER

def binary_classification_loss(self,
                               predicted_probs: tf.Tensor,
                               predictions: tf.Tensor, 
                               labels: tf.Tensor, 
                               pos_weight: float, 
                               neg_weight: float):
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


