import tensorflow as tf
from collections import namedtuple
from enum import Enum, auto
from typing import List

from .basic import mlp
from utils.loss_utils import binary_classification_loss, f1_score_loss
from utils.constants import ONE_HALF, SMALL_NUMBER


# Tuples to store output types
ClassificationOutput = namedtuple('ClassificationOutput', ['logits', 'prediction_probs', 'predictions', 'loss', 'accuracy', 'f1_score', 'precision', 'recall'])
RegressionOutput = namedtuple('RegressionOutput', ['predictions', 'loss'])


# Enum to denote output layer type
class OutputType(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()


def compute_regression_output(model_output: tf.Tensor, expected_output: tf.Tensor) -> RegressionOutput:
    """
    Uses the model output and expected output to compute the prediction and loss for this regression     task.
    """
    return RegressionOutput(predictions=model_output,
                            loss=tf.reduce_mean(tf.square(model_output - expected_output)))


def compute_binary_classification_output(model_output: tf.Tensor, labels: tf.Tensor, false_pos_weight: float, false_neg_weight: float, mode: str) -> ClassificationOutput:
    """
    Uses the model output and expected output to compute the classification output values for the
    given binary classification task.

    Args:
        model_output: A [B, 1] tensor containing the model outputs for each batch sample
        labels: A [B] float tensor with the correct labels
        false_pos_weight: Weight penalty for false positives
        false_neg_weight: Weight penalty for false negatives
    """
    logits = model_output
    predicted_probs = tf.math.sigmoid(logits)
    predictions = tf.cast(predicted_probs > ONE_HALF, dtype=tf.float32)

    mode = mode.lower()
    if mode == 'cross-entropy':
        loss = binary_classification_loss(predicted_probs=predicted_probs,
                                          predictions=predictions,
                                          labels=labels,
                                          pos_weight=false_pos_weight,
                                          neg_weight=false_neg_weight)
    elif mode == 'f1':
        loss = f1_score_loss(predicted_probs=predicted_probs, labels=labels)
    else:
        raise ValueError(f'Unknown loss mode {mode}')

    accuracy = tf.reduce_mean(1.0 - tf.abs(predictions - labels))

    # Compute F1 score (harmonic mean of precision and recall)
    true_positives = tf.reduce_sum(predictions * labels)
    false_positives = tf.reduce_sum(predictions * (1.0 - labels))
    false_negatives = tf.reduce_sum((1.0 - predictions) * labels)

    precision = true_positives / (true_positives + false_positives + SMALL_NUMBER)
    recall = true_positives / (true_positives + false_negatives + SMALL_NUMBER)

    f1_score = 2 * (precision * recall) / (precision + recall + SMALL_NUMBER)

    return ClassificationOutput(logits=logits,
                                prediction_probs=predicted_probs,
                                predictions=predictions,
                                loss=loss,
                                accuracy=accuracy,
                                f1_score=f1_score,
                                precision=precision,
                                recall=recall)
