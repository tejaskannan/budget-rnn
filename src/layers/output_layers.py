import tensorflow as tf
from collections import namedtuple
from enum import Enum, auto
from typing import List

from .basic import mlp
from utils.loss_utils import binary_classification_loss, f1_score_loss
from utils.constants import ONE_HALF, SMALL_NUMBER
from utils.tfutils import tf_f1_score


# Tuples to store output types
ClassificationOutput = namedtuple('ClassificationOutput', ['logits', 'prediction_probs', 'predictions', 'accuracy', 'f1_score'])
RegressionOutput = namedtuple('RegressionOutput', ['predictions'])


# Enum to denote output layer type
class OutputType(Enum):
    CLASSIFICATION = auto()
    REGRESSION = auto()


def compute_binary_classification_output(model_output: tf.Tensor, labels: tf.Tensor) -> ClassificationOutput:
    """
    Uses the model output and expected output to compute the classification output values for the
    given binary classification task.

    Args:
        model_output: A [B, 1] tensor containing the model outputs (logits) for each batch sample
        labels: A [B] float tensor with the correct labels
    """
    logits = model_output
    predicted_probs = tf.math.sigmoid(logits)
    predictions = tf.cast(predicted_probs > ONE_HALF, dtype=tf.float32)

    # Compute the batch-wise accuracy
    accuracy = tf.reduce_mean(1.0 - tf.abs(predictions - labels))

    # Compute F1 score (harmonic mean of precision and recall)
    f1_score = tf_f1_score(predictions, labels)

    return ClassificationOutput(logits=logits,
                                prediction_probs=predicted_probs,
                                predictions=predictions,
                                accuracy=accuracy,
                                f1_score=f1_score)
