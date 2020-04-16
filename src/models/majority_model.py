import tensorflow as tf
import numpy as np
import time

from enum import Enum, auto
from collections import defaultdict
from typing import Optional, Dict, List, Any, DefaultDict, Iterable

from models.standard_model import StandardModel, StandardModelType
from layers.basic import mlp, pool_sequence
from layers.output_layers import OutputType, compute_binary_classification_output, compute_multi_classification_output
from layers.embedding_layer import embedding_layer
from dataset.dataset import Dataset, DataSeries
from utils.hyperparameters import HyperParameters
from utils.misc import sample_sequence_batch
from utils.tfutils import pool_rnn_outputs, get_activation, tf_rnn_cell, get_rnn_state, majority_vote
from utils.constants import ACCURACY, ONE_HALF, OUTPUT, INPUTS, LOSS, PREDICTION, F1_SCORE, LOGITS, NODE_REGEX_FORMAT
from utils.constants import INPUT_SHAPE, NUM_OUTPUT_FEATURES, INPUT_SCALER, OUTPUT_SCALER, SEQ_LENGTH, DROPOUT_KEEP_RATE, MODEL, INPUT_NOISE
from utils.rnn_utils import get_prediction_name
from utils.constants import LABEL_MAP, REV_LABEL_MAP, NUM_CLASSES
from utils.testing_utils import ClassificationMetric, RegressionMetric, get_binary_classification_metric, get_regression_metric, ALL_LATENCY, get_multi_classification_metric
from utils.loss_utils import binary_classification_loss, f1_score_loss
from utils.np_utils import np_majority


# Layer name constants
EMBEDDING_LAYER_NAME = 'embedding-layer'
TRANSFORM_LAYER_NAME = 'transform-layer'
OUTPUT_LAYER_NAME = 'output-layer'
RNN_NAME = 'rnn'
BIRNN_NAME = 'birnn'


class MajorityModel(StandardModel):

    def __init__(self, hyper_parameters: HyperParameters, save_folder: str, is_train: bool):
        super().__init__(hyper_parameters, save_folder, is_train)

        model_type = self.hypers.model_params['model_type'].upper()
        self._model_type = StandardModelType[model_type]

        self.name = 'MAJORITY-' + model_type

    def _make_model(self, is_train: bool):
        # Embed the input sequence into a [B, T, D] tensor
        input_sequence = embedding_layer(inputs=self._placeholders[INPUTS],
                                         units=self.hypers.model_params['state_size'],
                                         dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                         use_conv=self.hypers.model_params['use_conv_embedding'],
                                         params=self.hypers.model_params['embedding_layer_params'],
                                         seq_length=self.metadata[SEQ_LENGTH],
                                         input_shape=self.metadata[INPUT_SHAPE],
                                         name_prefix=EMBEDDING_LAYER_NAME)

        # Apply the transformation layer
        if self.model_type == StandardModelType.NBOW:
            # Apply the MLP transformation. Outputs a [B, T, D] tensor
            transformed = mlp(inputs=input_sequence,
                              output_size=self.hypers.model_params['state_size'],
                              hidden_sizes=self.hypers.model_params['mlp_hidden_units'],
                              activations=self.hypers.model_params['mlp_activation'],
                              dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                              should_activate_final=True,
                              should_bias_final=True,
                              should_dropout_final=True,
                              name=TRANSFORM_LAYER_NAME)
        elif self.model_type == StandardModelType.CNN:
            # Apply the 1 dimensional CNN transformation. Outputs a [B, T, D] tensor
            transformed = tf.layers.conv1d(inputs=input_sequence,
                                           filters=self.hypers.model_params['state_size'],
                                           kernel_size=self.hypers.model_params['cnn_kernel_size'],
                                           strides=self.hypers.model_params['cnn_strides'],
                                           padding='same',
                                           activation=get_activation(self.hypers.model_params['cnn_activation']),
                                           use_bias=True,
                                           kernel_initializer=tf.glorot_uniform_initializer(),
                                           name=TRANSFORM_LAYER_NAME)
            # Apply dropout, [B, T, D]
            transformed = tf.nn.dropout(transformed, keep_rate=self._placeholders[DROPOUT_KEEP_RATE])
        elif self.model_type == StandardModelType.RNN:
            cell = tf_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                               num_units=self.hypers.model_params['state_size'],
                               activation=self.hypers.model_params['rnn_activation'],
                               layers=self.hypers.model_params['rnn_layers'],
                               dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                               name_prefix=TRANSFORM_LAYER_NAME)

            initial_state = cell.zero_state(batch_size=tf.shape(input_sequence)[0], dtype=tf.float32)
            transformed, state = tf.nn.dynamic_rnn(cell=cell,
                                                   inputs=input_sequence,
                                                   initial_state=initial_state,
                                                   dtype=tf.float32,
                                                   scope=RNN_NAME)
        elif self.model_type == StandardModelType.BIRNN:
            fw_cell = tf_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                  num_units=self.hypers.model_params['state_size'],
                                  activation=self.hypers.model_params['rnn_activation'],
                                  layers=self.hypers.model_params['rnn_layers'],
                                  dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                  name_prefix=f'{TRANSFORM_LAYER_NAME}-fw')
            
            bw_cell = tf_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                  num_units=self.hypers.model_params['state_size'],
                                  activation=self.hypers.model_params['rnn_activation'],
                                  layers=self.hypers.model_params['rnn_layers'],
                                  dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                  name_prefix=f'{TRANSFORM_LAYER_NAME}-bw')

            batch_size = tf.shape(input_sequence)[0]
            fw_initial_state = fw_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            bw_initial_state = bw_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            transformed, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                 cell_bw=bw_cell,
                                                                 inputs=input_sequence,
                                                                 initial_state_fw=fw_initial_state,
                                                                 initial_state_bw=bw_initial_state,
                                                                 dtype=tf.float32,
                                                                 scope=BIRNN_NAME)
            # Concatenate forward and backward states  # [B, T, 2 * D]
            transformed = tf.concat(transformed, axis=-1)
        else:
            raise ValueError(f'Unknown transformation type: {self.model_type}')

        # Create the output layer. Outputs a [B, T, M] tensor.
        output_size = self.metadata[NUM_OUTPUT_FEATURES] if self.output_type != OutputType.MULTI_CLASSIFICATION else self.metadata[NUM_CLASSES]
        output = mlp(inputs=transformed,
                     output_size=output_size,
                     hidden_sizes=self.hypers.model_params['output_hidden_units'],
                     activations=self.hypers.model_params['output_hidden_activation'],
                     dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                     name=OUTPUT_LAYER_NAME)
        
        # Create [B, T, M] tensor of the expected output to align with the sequence elements
        expected_output = tf.tile(tf.expand_dims(self._placeholders[OUTPUT], axis=1),
                                  multiples=(1, tf.shape(output)[1], 1))
        output_shape = tf.shape(output)

        if self.output_type == OutputType.BINARY_CLASSIFICATION:
            reshaped_expected = tf.reshape(expected_output, shape=(-1, 1))  # [B * T, D]
            reshaped_output = tf.reshape(output, shape=(-1, output_shape[-1]))  # [B * T, D]

            classification_output = compute_binary_classification_output(model_output=reshaped_output,
                                                                         labels=reshaped_expected)

            self._ops[LOGITS] = tf.reshape(classification_output.logits, shape=(-1, output_shape[1], output_shape[2]))  # [B, T, M]
            self._ops[PREDICTION] = majority_vote(self._ops[LOGITS])  # [B]
            self._ops[ACCURACY] = classification_output.accuracy  # Scalar
            self._ops[F1_SCORE] = classification_output.f1_score  # Scalar
        elif self.output_type == OutputType.MULTI_CLASSIFICATION:
            reshaped_expected = tf.reshape(expected_output, shape=(-1, 1))  # [B * T, D]
            reshaped_output = tf.reshape(output, shape=(-1, output_shape[-1]))  # [B * T, D]

            classification_output = compute_multi_classification_output(model_output=reshaped_output,
                                                                        labels=reshaped_expected)
            self._ops[LOGITS] = tf.reshape(classification_output.logits, shape=(-1, output_shape[1], output_shape[2]))  # [B, T, M]
            self._ops[PREDICTION] = majority_vote(self._ops[LOGITS])  # [B]
            self._ops[ACCURACY] = classification_output.accuracy  # Scalar
            self._ops[F1_SCORE] = classification_output.f1_score  # Scalar
        else:
            self._ops[PREDICTION] = output

    def make_loss(self):
        # Reshape expected output to account for sequence length
        expected_output = tf.tile(tf.expand_dims(self._placeholders[OUTPUT], axis=1),
                                  multiples=(1, tf.shape(self._placeholders[INPUTS])[1], 1))
        predictions = self._ops[PREDICTION]

        if self.output_type == OutputType.BINARY_CLASSIFICATION:
            loss_mode = self.hypers.model_params['loss_mode'].lower()

            logits = self._ops[LOGITS]
            predicted_probs = tf.math.sigmoid(logits)

            if loss_mode in ('binary-cross-entropy', 'binary_cross_entropy'):
                sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=expected_output,
                                                                      logits=logits)
                self._ops[LOSS] = tf.reduce_mean(sample_loss)
            elif loss_mode in ('f1', 'f1-score', 'f1_score'):
                self._ops[LOSS] = f1_score_loss(predicted_probs=predicted_probs,
                                                labels=expected_output)
            else:
                raise ValueError(f'Unknown loss mode: {loss_mode}')

        elif self.output_type == OutputType.MULTI_CLASSIFICATION:
            logits = tf.reshape(self._ops[LOGITS], shape=(-1, tf.shape(self._ops[LOGITS])[-1]))
            labels = tf.reshape(expected_output, shape=(-1, ))

            sample_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            self._ops[LOSS] = tf.reduce_mean(sample_loss)
        else:
            self._ops[LOSS] = tf.reduce_mean(tf.square(predictions - expected_output))

    def predict_classification(self, test_batch_generator: Iterable[Any],
                               batch_size: int,
                               max_num_batches: Optional[int],
                               flops_dict: Dict[str, int]) -> DefaultDict[str, Dict[str, Any]]:
        sample_frac = self.hypers.model_params['sample_frac']
        num_sequences = int(1.0 / sample_frac)
        samples_per_seq = int(self.metadata[SEQ_LENGTH] * sample_frac)

        predictions_dict: DefaultDict[str, List[np.ndarray]] = defaultdict(list)
        labels_list: List[np.ndarray] = []
        latencies: List[float] = []

        for batch_num, batch in enumerate(test_batch_generator):
            feed_dict = self.batch_to_feed_dict(batch, is_train=False)

            start = time.time()
            logits = self.sess.run(self._ops[LOGITS], feed_dict=feed_dict)
            elapsed = time.time() - start

            labels_list.append(np.vstack(batch[OUTPUT]))
            latencies.append(elapsed)

            for level in range(num_sequences):
                start, end = level * samples_per_seq, (level + 1) * samples_per_seq

                sub_sequence_logits = logits[:, start:end, :]
                level_predictions = np_majority(sub_sequence_logits)

                predictions_dict[get_prediction_name(level)].append(np.vstack(level_predictions))

            if (max_num_batches is not None) and (batch_num >= max_num_batches):
                break

        labels = np.vstack(labels_list)
        avg_latency = np.average(latencies[1:])

        result: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
        flops = 0
        for level in range(num_sequences):
            flops += sample_frac * flops_dict[PREDICTION]

            level_name = get_prediction_name(level)
            predictions = np.vstack(predictions_dict[level_name])

            for metric_name in ClassificationMetric:
                if self.output_type == OutputType.BINARY_CLASSIFICATION:
                    metric_value = get_binary_classification_metric(metric_name, predictions, labels, avg_latency, 1, flops)
                else:
                    metric_value = get_multi_classification_metric(metric_name, predictions, labels, avg_latency, 1, flops, self.metadata[NUM_CLASSES])

                result[level_name][metric_name.name] = metric_value

            result[level_name][ALL_LATENCY] = latencies[1:]

        return result
