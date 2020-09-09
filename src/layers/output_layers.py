import tensorflow as tf
from collections import namedtuple
from enum import Enum, auto
from typing import List

from utils.constants import ONE_HALF, SMALL_NUMBER


# Tuples to store output types
ClassificationOutput = namedtuple('ClassificationOutput', ['logits', 'prediction_probs', 'predictions', 'accuracy'])
RegressionOutput = namedtuple('RegressionOutput', ['predictions'])


# Enum to denote output layer type
class OutputType(Enum):
    BINARY_CLASSIFICATION = auto()
    MULTI_CLASSIFICATION = auto()
    REGRESSION = auto()


def is_classification(output_type: OutputType) -> bool:
    return output_type in (OutputType.BINARY_CLASSIFICATION, OutputType.MULTI_CLASSIFICATION)


def compute_binary_classification_output(model_output: tf.Tensor, labels: tf.Tensor) -> ClassificationOutput:
    """
    Uses the model output and expected output to compute the classification output values for the
    given binary classification task.

    Args:
        model_output: A [B, 1] tensor containing the model outputs (logits) for each batch sample
        labels: A [B, 1] float tensor with the correct labels
    """
    logits = model_output
    predicted_probs = tf.math.sigmoid(logits)
    predictions = tf.cast(predicted_probs > ONE_HALF, dtype=tf.float32)

    # Compute the batch-wise accuracy
    accuracy = tf.reduce_mean(1.0 - tf.abs(predictions - labels))

    return ClassificationOutput(logits=logits,
                                prediction_probs=predicted_probs,
                                predictions=predictions,
                                accuracy=accuracy)


def compute_multi_classification_output(model_output: tf.Tensor, labels: tf.Tensor) -> ClassificationOutput:
    """
    Uses the model output to compute the multi-class classification output for a given task.

    Args:
        model_output: A [B, K] or [B, T, K] tensor containing the logits for each batch sample (B) and output class (K)
        labels: A [B, 1] or [B, T, 1] int tensor with the expected labels.
    """
    logits = model_output  # [B, K] / [B, T, K]
    predicted_probs = tf.nn.softmax(logits, axis=-1)  # [B, K]  / [B, T, K]
    predictions = tf.math.argmax(predicted_probs, axis=-1, output_type=labels.dtype)  # [B] / [B, T]

    # Compute the batch-wise accuracy
    correct = tf.cast(tf.equal(predictions, tf.squeeze(labels, axis=-1)), dtype=tf.float32)
    accuracy = tf.reduce_mean(correct)

    return ClassificationOutput(logits=logits,
                                prediction_probs=predicted_probs,
                                predictions=predictions,
                                accuracy=accuracy)
