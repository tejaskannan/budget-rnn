import tensorflow as tf
import numpy as np
import time
from enum import Enum, auto
from collections import defaultdict
from typing import Optional, Dict, List, Any, DefaultDict, Iterable

from models.tf_model import TFModel
from layers.cells.cell_factory import make_rnn_cell, CellClass, CellType
from layers.dense import mlp, dense
from layers.output_layers import OutputType, compute_binary_classification_output, compute_multi_classification_output
from dataset.dataset import Dataset, DataSeries
from utils.hyperparameters import HyperParameters
from utils.misc import sample_sequence_batch, batch_sample_noise
from utils.tfutils import get_activation, successive_pooling, apply_noise
from utils.sequence_model_utils import SequenceModelType
from utils.constants import ACCURACY, OUTPUT, INPUTS, LOSS, PREDICTION, LOGITS, SMALL_NUMBER, LEAK_RATE, PHASE_GATES
from utils.constants import INPUT_SHAPE, NUM_OUTPUT_FEATURES, SEQ_LENGTH, DROPOUT_KEEP_RATE, MODEL, NUM_CLASSES, ACTIVATION_NOISE
from utils.constants import EMBEDDING_NAME, TRANSFORM_NAME, AGGREGATION_NAME, OUTPUT_LAYER_NAME, RNN_NAME, SKIP_GATES, RNN_CELL_NAME
from utils.testing_utils import ClassificationMetric, RegressionMetric, get_binary_classification_metric, get_regression_metric, get_multi_classification_metric
from utils.loss_utils import binary_classification_loss, get_loss_weights
from .base_model import Model


class StandardModel(TFModel):

    def __init__(self, hyper_parameters: HyperParameters, save_folder: str, is_train: bool):
        super().__init__(hyper_parameters, save_folder, is_train)

        model_type = self.hypers.model_params['model_type'].upper()
        self._model_type = SequenceModelType[model_type]

        self.name = model_type

    @property
    def model_type(self) -> SequenceModelType:
        return self._model_type

    @property
    def seq_length(self) -> int:
        return self.metadata[SEQ_LENGTH]

    @property
    def prediction_op_name(self) -> str:
        return PREDICTION

    @property
    def output_op_name(self) -> str:
        return self.prediction_op_name

    @property
    def num_output_features(self) -> int:
        if self.output_type == OutputType.MULTI_CLASSIFICATION:
            return int(self.metadata[NUM_CLASSES])
        return int(self.metadata[NUM_OUTPUT_FEATURES])

    def batch_to_feed_dict(self, batch: Dict[str, List[Any]], is_train: bool, epoch_num: int) -> Dict[tf.Tensor, np.ndarray]:
        dropout = self.hypers.dropout_keep_rate if is_train else 1.0
        activation_noise = self.hypers.input_noise if is_train else 0.0
        input_batch = np.array(batch[INPUTS])
        output_batch = np.array(batch[OUTPUT])

        if input_batch.shape[1] == 1:
            input_batch = np.squeeze(input_batch, axis=1)

        input_shape = self.metadata[INPUT_SHAPE]
        num_output_features = self.metadata[NUM_OUTPUT_FEATURES]
        seq_length = self.metadata[SEQ_LENGTH]

        feed_dict = {
            self._placeholders[INPUTS]: input_batch,
            self._placeholders[OUTPUT]: output_batch.reshape(-1, num_output_features),
            self._placeholders[DROPOUT_KEEP_RATE]: dropout,
            self._placeholders[ACTIVATION_NOISE]: activation_noise
        }

        if self.model_type == SequenceModelType.PHASED_RNN:
            feed_dict[self._placeholders[LEAK_RATE]] = self.hypers.model_params['leak_rate'] if is_train else 0.0

        return feed_dict

    def make_placeholders(self, is_frozen: bool = False):
        input_features_shape = self.metadata[INPUT_SHAPE]
        num_output_features = self.metadata[NUM_OUTPUT_FEATURES]
        seq_length = self.metadata[SEQ_LENGTH]

        input_shape = (None, seq_length) + input_features_shape
        output_dtype = tf.int32 if self.output_type == OutputType.MULTI_CLASSIFICATION else tf.float32

        if not is_frozen:
            self._placeholders[INPUTS] = tf.placeholder(shape=input_shape,
                                                        dtype=tf.float32,
                                                        name=INPUTS)
            self._placeholders[OUTPUT] = tf.placeholder(shape=(None, num_output_features),
                                                        dtype=output_dtype,
                                                        name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = tf.placeholder(shape=(),
                                                                   dtype=tf.float32,
                                                                   name=DROPOUT_KEEP_RATE)
            self._placeholders[ACTIVATION_NOISE] = tf.placeholder(shape=(),
                                                                  dtype=tf.float32,
                                                                  name=ACTIVATION_NOISE)
            # Phased RNNs have an extra leak rate placeholder
            if self.model_type == SequenceModelType.PHASED_RNN:
                self._placeholders[LEAK_RATE] = tf.placeholder(shape=(),
                                                               dtype=tf.float32,
                                                               name=LEAK_RATE)
        else:
            self._placeholders[INPUTS] = tf.ones(shape=(1,) + input_shape[1:], dtype=tf.float32, name=INPUTS)
            self._placeholders[OUTPUT] = tf.ones(shape=(1, num_output_features), dtype=output_dtype, name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = tf.ones(shape=(), dtype=tf.float32, name=DROPOUT_KEEP_RATE)
            self._placeholders[ACTIVATION_NOISE] = tf.zeros(shape=(), dtype=tf.float32, name=ACTIVATION_NOISE)

            if self.model_type == SequenceModelType.PHASED_RNN:
                self._placeholders[LEAK_RATE] = tf.ones(shape=(), dtype=tf.float32, name=LEAK_RATE)

    def make_model(self, is_train: bool):
        with tf.variable_scope(MODEL, reuse=tf.AUTO_REUSE):
            self._make_model(is_train)

    def _make_model(self, is_train: bool):
        """
        Builds the computation graph for this model.
        """
        state_size = self.hypers.model_params['state_size']
        batch_size = tf.shape(self._placeholders[INPUTS])[0]
        activation_noise = self._placeholders[ACTIVATION_NOISE]
        dropout_keep_rate = self._placeholders[DROPOUT_KEEP_RATE]

        # Apply input noise
        inputs = apply_noise(self._placeholders[INPUTS], scale=activation_noise)

        # Embed the input sequence into a [B, T, D] tensor
        embeddings, _ = dense(inputs=inputs,
                              units=state_size,
                              activation=self.hypers.model_params['embedding_activation'],
                              use_bias=True,
                              activation_noise=activation_noise,
                              name=EMBEDDING_NAME)

        # Apply the transformation layer. The output is a [B, T, D] tensor of transformed inputs for each model type.
        if self.model_type == SequenceModelType.NBOW:
            # Apply the MLP transformation. Result is a [B, T, D] tensor
            transformed, _ = mlp(inputs=embeddings,
                                 output_size=state_size,
                                 hidden_sizes=self.hypers.model_params['mlp_hidden_units'],
                                 activations=self.hypers.model_params['mlp_activation'],
                                 dropout_keep_rate=dropout_keep_rate,
                                 activation_noise=activation_noise,
                                 should_activate_final=True,
                                 should_bias_final=True,
                                 should_dropout_final=True,
                                 name=TRANSFORM_NAME)

            # Compute weights for aggregation layer, [B, T, 1]
            aggregation_weights, _ = dense(inputs=transformed,
                                           units=1,
                                           activation='sigmoid',
                                           activation_noise=activation_noise,
                                           use_bias=True,
                                           name=AGGREGATION_NAME)

            # Pool the data in a successive fashion, [B, T, D]
            transformed = successive_pooling(inputs=transformed,
                                             aggregation_weights=aggregation_weights,
                                             name='{0}-pool'.format(AGGREGATION_NAME),
                                             seq_length=self.metadata[SEQ_LENGTH])
        elif self.model_type == SequenceModelType.RNN:
            cell = make_rnn_cell(cell_class=CellClass.STANDARD,
                                 cell_type=CellType[self.hypers.model_params['rnn_cell_type'].upper()],
                                 units=state_size,
                                 activation=self.hypers.model_params['rnn_activation'],
                                 recurrent_noise=activation_noise,
                                 name=RNN_CELL_NAME)

            initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            rnn_outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                                   inputs=embeddings,
                                                   initial_state=initial_state,
                                                   dtype=tf.float32,
                                                   scope=TRANSFORM_NAME)
            transformed = rnn_outputs  # [B, T, D]
        elif self.model_type == SequenceModelType.SKIP_RNN:
            cell = make_rnn_cell(cell_class=CellClass.SKIP,
                                 cell_type=CellType[self.hypers.model_params['rnn_cell_type'].upper()],
                                 units=state_size,
                                 activation=self.hypers.model_params['rnn_activation'],
                                 recurrent_noise=activation_noise,
                                 name=RNN_CELL_NAME)

            initial_state = cell.get_initial_state(inputs=embeddings,
                                                   batch_size=batch_size,
                                                   dtype=tf.float32)
            # Apply RNN
            rnn_outputs, states = tf.nn.dynamic_rnn(cell=cell,
                                                    inputs=embeddings,
                                                    initial_state=initial_state,
                                                    dtype=tf.float32,
                                                    scope=TRANSFORM_NAME)
            transformed = rnn_outputs.output  # [B, T, D]
            self._ops[SKIP_GATES] = tf.squeeze(rnn_outputs.state_update_gate, axis=-1)  # [B, T]
        elif self.model_type == SequenceModelType.PHASED_RNN:
            period_init = self.metadata[SEQ_LENGTH]

            cell = make_rnn_cell(cell_class=CellClass.PHASED,
                                 cell_type=CellType[self.hypers.model_params['rnn_cell_type'].upper()],
                                 units=state_size,
                                 activation=self.hypers.model_params['rnn_activation'],
                                 recurrent_noise=activation_noise,
                                 on_fraction=self.hypers.model_params['on_fraction'],
                                 period_init=period_init,
                                 leak_rate=self.placeholders[LEAK_RATE],
                                 name=RNN_CELL_NAME)

            initial_state = cell.get_initial_state(inputs=embeddings,
                                                   batch_size=batch_size,
                                                   dtype=tf.float32)

            rnn_outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                                   inputs=embeddings,
                                                   initial_state=initial_state,
                                                   dtype=tf.float32,
                                                   scope=TRANSFORM_NAME)
            transformed = rnn_outputs.output  # [B, T, D]
            self._ops[PHASE_GATES] = tf.squeeze(rnn_outputs.time_gate, axis=-1)  # [B, T]
        else:
            raise ValueError('Unknown standard model: {0}'.format(self.model_type))

        # Reshape the output to match the sequence length. The output is tiled along the sequence length
        # automatically via broadcasting rules.
        if self.hypers.model_params.get('has_single_output', False):
            transformed = transformed[:, -1, :]  # Take the final transformed state, [B, D]
            expected_output = self._placeholders[OUTPUT]
        else:
            expected_output = tf.expand_dims(self._placeholders[OUTPUT], axis=-1)  # [B, 1, 1]

        # Create the output layer, result is a [B, T, C] tensor or a [B, C] tensor depending on the output type
        output_size = self.metadata[NUM_OUTPUT_FEATURES] if self.output_type != OutputType.MULTI_CLASSIFICATION else self.metadata[NUM_CLASSES]
        output, _ = mlp(inputs=transformed,
                        output_size=self.num_output_features,
                        hidden_sizes=self.hypers.model_params['output_hidden_units'],
                        activations=self.hypers.model_params['output_hidden_activation'],
                        dropout_keep_rate=dropout_keep_rate,
                        activation_noise=activation_noise,
                        should_bias_final=True,
                        should_activate_final=False,
                        should_dropout_final=False,
                        name=OUTPUT_LAYER_NAME)

        if self.output_type == OutputType.BINARY_CLASSIFICATION:
            classification_output = compute_binary_classification_output(model_output=output,
                                                                         labels=expected_output)
            self._ops[LOGITS] = classification_output.logits
            self._ops[PREDICTION] = classification_output.predictions
            self._ops[ACCURACY] = classification_output.accuracy
        elif self.output_type == OutputType.MULTI_CLASSIFICATION:
            classification_output = compute_multi_classification_output(model_output=output,
                                                                        labels=expected_output)
            self._ops[LOGITS] = classification_output.logits
            self._ops[PREDICTION] = classification_output.predictions
            self._ops[ACCURACY] = classification_output.accuracy
        else:
            self._ops[PREDICTION] = output

    def make_loss(self):
        # Tile the output along all sequence elements. This is necessary for the sparse softmax
        # cross entropy function. We only do this for multiple output models.
        seq_length = self.metadata[SEQ_LENGTH]
        has_single_output = self.hypers.model_params.get('has_single_output', False)

        if has_single_output:
            expected_output = self._placeholders[OUTPUT]
        else:
            expected_output = tf.expand_dims(self._placeholders[OUTPUT], axis=-1)  # [B, 1, 1]
            expected_output = tf.tile(expected_output, multiples=(1, seq_length, 1))  # [B, T, 1]

        predictions = self._ops[PREDICTION]  # [B, T, C] or [B, C] depending on the output type

        # Create the loss weights
        loss_weights = get_loss_weights(n=seq_length, mode=self.hypers.model_params.get('loss_weights'))

        # Expand for later broadcasting
        loss_weights = np.expand_dims(loss_weights, axis=0)  # [1, T]

        if self.output_type == OutputType.BINARY_CLASSIFICATION:
            logits = self._ops[LOGITS]
            sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=expected_output,
                                                                  logits=logits)
        elif self.output_type == OutputType.MULTI_CLASSIFICATION:
            logits = self._ops[LOGITS]
            labels = tf.squeeze(expected_output, axis=-1)  # [B, T] or [B] depending on the output type

            sample_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        else:
            sample_loss = tf.square(predictions - expected_output)

        # Compute weighted average over sequence elements if the model has multiple outputs
        if not has_single_output:
            sample_loss = tf.reduce_sum(sample_loss * loss_weights, axis=-1)

        # Average loss over the batch
        self._ops[LOSS] = tf.reduce_mean(sample_loss)

        # If we have a skip RNN, then we apply the L2 update penalty
        if self.model_type == SequenceModelType.SKIP_RNN:
            skip_gates = self._ops[SKIP_GATES]  # [B, T]
            target_updates = self.hypers.model_params['target_updates']

            # L2 penalty for deviation from target
            update_penalty = tf.square(tf.reduce_sum(skip_gates, axis=-1) - target_updates)  # [B]
            update_loss = tf.reduce_mean(update_penalty)  # Average over batch to get scalar loss

            self._ops[LOSS] += self.hypers.model_params['update_loss_weight'] * update_loss

    def predict_classification(self, test_batch_generator: Iterable[Any],
                               batch_size: int,
                               max_num_batches: Optional[int]) -> DefaultDict[str, Dict[str, Any]]:
        predictions_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        skip_gates_list: List[np.ndarray] = []  # Only used for Skip RNN models
        phase_gates_list: List[np.ndarray] = []  # Only used for Phased RNN models

        ops_to_run = [self.prediction_op_name, SKIP_GATES, PHASE_GATES]

        for batch_num, batch in enumerate(test_batch_generator):
            if max_num_batches is not None and batch_num >= max_num_batches:
                break

            feed_dict = self.batch_to_feed_dict(batch, is_train=False, epoch_num=0)
            batch_result = self.execute(ops=ops_to_run, feed_dict=feed_dict)

            prediction = batch_result[self.prediction_op_name]

            if batch_result.get(SKIP_GATES) is not None:
                skip_gates_list.append(batch_result[SKIP_GATES])

            if batch_result.get(PHASE_GATES) is not None:
                phase_gates_list.append(batch_result[PHASE_GATES])
                print(batch_result[PHASE_GATES])

            labels_list.append(np.vstack(batch[OUTPUT]))
            predictions_list.append(np.vstack(prediction))

        predictions = np.vstack(predictions_list)  # [B, T] or [B] depending on the output type
        labels = np.squeeze(np.vstack(labels_list), axis=-1)  # [B]

        result: DefaultDict[str, Dict[str, float]] = defaultdict(dict)

        if self.hypers.model_params.get('has_single_output', False):
            predictions = np.squeeze(predictions, axis=-1)  # [B]

            for metric_name in ClassificationMetric:
                if self.output_type == OutputType.BINARY_CLASSIFICATION:
                    metric_value = get_binary_classification_metric(metric_name, predictions, labels)
                else:
                    metric_value = get_multi_classification_metric(metric_name, predictions, labels, self.metadata[NUM_CLASSES])

                result[PREDICTION][metric_name.name] = metric_value

            # Add in the average number of updates for Skip RNNs. These always have one output, so
            # we only need this case in the has_single_output == True case.
            if self.model_type == SequenceModelType.SKIP_RNN:
                skip_gates = np.concatenate(skip_gates_list, axis=0)  # [B, T]
                num_updates = np.sum(skip_gates, axis=-1)  # [B]

                result[PREDICTION]['AVG_UPDATES'] = float(np.average(num_updates))
                result[PREDICTION]['STD_UPDATES'] = float(np.std(num_updates))
            
            if self.model_type == SequenceModelType.PHASED_RNN:
                phase_gates = np.concatenate(phase_gates_list, axis=0)  # [B, T]
                num_updates = np.count_nonzero(phase_gates, axis=-1)  # [B]

                result[PREDICTION]['AVG_UPDATES'] = float(np.average(num_updates))
        else:
            for i in range(self.metadata[SEQ_LENGTH]):
                level_name = '{0}_{1}'.format(PREDICTION, i)

                for metric_name in ClassificationMetric:
                    if self.output_type == OutputType.BINARY_CLASSIFICATION:
                        metric_value = get_binary_classification_metric(metric_name, predictions[:, i], labels)
                    else:
                        metric_value = get_multi_classification_metric(metric_name, predictions[:, i], labels, self.metadata[NUM_CLASSES])

                    result[level_name][metric_name.name] = metric_value

        return result
