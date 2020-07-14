import tensorflow as tf
import numpy as np
import time
from enum import Enum, auto
from collections import defaultdict
from typing import Optional, Dict, List, Any, DefaultDict, Iterable

from models.tf_model import TFModel
from layers.rnn import dynamic_rnn
from layers.cells.cells import make_rnn_cell
from layers.basic import mlp, pool_sequence, dense
from layers.output_layers import OutputType, compute_binary_classification_output, compute_multi_classification_output
from dataset.dataset import Dataset, DataSeries
from utils.hyperparameters import HyperParameters
from utils.misc import sample_sequence_batch, batch_sample_noise
from utils.tfutils import pool_rnn_outputs, get_activation, tf_rnn_cell, get_rnn_state, successive_pooling
from utils.constants import ACCURACY, OUTPUT, INPUTS, LOSS, PREDICTION, F1_SCORE, LOGITS, NODE_REGEX_FORMAT
from utils.constants import INPUT_SHAPE, NUM_OUTPUT_FEATURES, SEQ_LENGTH, DROPOUT_KEEP_RATE, MODEL, NUM_CLASSES
from utils.constants import AGGREGATE_SEED, TRANSFORM_SEED, OUTPUT_SEED, EMBEDDING_SEED, SMALL_NUMBER
from utils.testing_utils import ClassificationMetric, RegressionMetric, get_binary_classification_metric, get_regression_metric, ALL_LATENCY, get_multi_classification_metric
from utils.loss_utils import binary_classification_loss, f1_score_loss
from utils.rnn_utils import get_backward_name
from .base_model import Model


# Layer name constants
EMBEDDING_LAYER_NAME = 'embedding-layer'
TRANSFORM_LAYER_NAME = 'transform-layer'
AGGREGATION_LAYER_NAME = 'aggregation-layer'
OUTPUT_LAYER_NAME = 'output-layer'
RNN_NAME = 'rnn'
BIRNN_NAME = 'birnn'


class StandardModelType(Enum):
    NBOW = auto()
    CNN = auto()
    RNN = auto()
    BIRNN = auto()


class StandardModel(TFModel):

    def __init__(self, hyper_parameters: HyperParameters, save_folder: str, is_train: bool):
        super().__init__(hyper_parameters, save_folder, is_train)

        model_type = self.hypers.model_params['model_type'].upper()
        self._model_type = StandardModelType[model_type]

        self.name = model_type

    @property
    def model_type(self) -> StandardModelType:
        return self._model_type

    @property
    def prediction_ops(self) -> List[str]:
        return [PREDICTION]

    @property
    def accuracy_op_names(self) -> List[str]:
        return [ACCURACY]

    @property
    def loss_op_names(self) -> List[str]:
        return [LOSS]

    @property
    def output_ops(self) -> List[str]:
        return self.prediction_ops

    def batch_to_feed_dict(self, batch: Dict[str, List[Any]], is_train: bool) -> Dict[tf.Tensor, np.ndarray]:
        dropout = self.hypers.dropout_keep_rate if is_train else 1.0
        input_batch = np.array(batch[INPUTS])
        output_batch = np.array(batch[OUTPUT])

        if input_batch.shape[1] == 1:
            input_batch = np.squeeze(input_batch, axis=1)

        input_shape = self.metadata[INPUT_SHAPE]
        num_output_features = self.metadata[NUM_OUTPUT_FEATURES]
        seq_length = self.metadata[SEQ_LENGTH]

        # Add noise to batch during training
        if is_train and self.hypers.batch_noise > SMALL_NUMBER:
            input_batch = batch_sample_noise(input_batch, noise_weight=self.hypers.batch_noise)

        feed_dict = {
            self._placeholders[INPUTS]: input_batch,
            self._placeholders[OUTPUT]: output_batch.reshape(-1, num_output_features),
            self._placeholders[DROPOUT_KEEP_RATE]: dropout
        }

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
        else:
            self._placeholders[INPUTS] = tf.ones(shape=(1,) + input_shape[1:], dtype=tf.float32, name=INPUTS)
            self._placeholders[OUTPUT] = tf.ones(shape=(1, num_output_features), dtype=output_dtype, name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = tf.ones(shape=(), dtype=tf.float32, name=DROPOUT_KEEP_RATE)

    def make_model(self, is_train: bool):
        with tf.variable_scope(MODEL, reuse=tf.AUTO_REUSE):
            self._make_model(is_train)

    def _make_model(self, is_train: bool):
        """
        Builds the comptuation graph based on the model type.
        """
        compression_fraction = self.hypers.model_params.get('compression_fraction')
        state_size = self.hypers.model_params['state_size']
        batch_size = tf.shape(self._placeholders[INPUTS])[0]

        # Embed the input sequence into a [B, T, D] tensor
        input_sequence, _ = dense(inputs=self._placeholders[INPUTS],
                                  units=state_size,
                                  activation=self.hypers.model_params['embedding_activation'],
                                  use_bias=True,
                                  name=EMBEDDING_LAYER_NAME,
                                  compression_fraction=compression_fraction,
                                  compression_seed=EMBEDDING_SEED)

        # Apply the transformation layer. The output is a [B, T, D] tensor of transformed inputs for each model type.
        if self.model_type == StandardModelType.NBOW:
            # Apply the MLP transformation. Result is a [B, T, D] tensor
            transformed, _ = mlp(inputs=input_sequence,
                                 output_size=state_size,
                                 hidden_sizes=self.hypers.model_params['mlp_hidden_units'],
                                 activations=self.hypers.model_params['mlp_activation'],
                                 dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                 should_activate_final=True,
                                 should_bias_final=True,
                                 should_dropout_final=True,
                                 name=TRANSFORM_LAYER_NAME,
                                 compression_fraction=compression_fraction,
                                 compression_seed=TRANSFORM_SEED)

            # Compute weights for aggregation layer, [B, T, 1]
            aggregation_weights, _ = dense(inputs=transformed,
                                           units=1,
                                           activation='sigmoid',
                                           use_bias=True,
                                           name=AGGREGATION_LAYER_NAME,
                                           compression_fraction=compression_fraction,
                                           compression_seed=AGGREGATE_SEED)

            # Pool the data in a successive fashion, [B, T, D]
            transformed = successive_pooling(inputs=transformed,
                                             aggregation_weights=aggregation_weights,
                                             name='{0}-pool'.format(AGGREGATION_LAYER_NAME),
                                             seq_length=self.metadata[SEQ_LENGTH])
        elif self.model_type == StandardModelType.RNN:
            # We either use a tensorflow cell or a custom RNN cell depending on whether we
            # are compressing the trainable parameters. The compressed cell uses the custom implementation.
            if compression_fraction is None:
                cell = tf_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                   num_units=state_size,
                                   activation=self.hypers.model_params['rnn_activation'],
                                   layers=self.hypers.model_params['rnn_layers'],
                                   name_prefix=TRANSFORM_LAYER_NAME)

                initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
                rnn_outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                                       inputs=input_sequence,
                                                       initial_state=initial_state,
                                                       dtype=tf.float32,
                                                       scope=RNN_NAME)
                transformed = rnn_outputs
            else:
                cell = make_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                     input_units=state_size,
                                     output_units=state_size,
                                     activation=self.hypers.model_params['rnn_activation'],
                                     num_layers=self.hypers.model_params['rnn_layers'],
                                     name=TRANSFORM_LAYER_NAME,
                                     compression_fraction=compression_fraction,
                                     compression_seed=TRANSFORM_SEED)

                initial_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)

                # Run RNN and collect outputs
                rnn_out = dynamic_rnn(cell=cell,
                                      inputs=input_sequence,
                                      previous_states=None,
                                      initial_state=initial_state,
                                      name=RNN_NAME,
                                      compression_fraction=compression_fraction)
                rnn_outputs = rnn_out.outputs
                rnn_states = rnn_out.states
                rnn_gates = rnn_out.gates

                transformed = rnn_outputs.stack()  # [T, B, D]
                transformed = tf.transpose(transformed, perm=[1, 0, 2])  # [B, T, D]
        elif self.model_type == StandardModelType.BIRNN:
            if compression_fraction is None:
                fw_cell = tf_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                      num_units=state_size,
                                      activation=self.hypers.model_params['rnn_activation'],
                                      layers=self.hypers.model_params['rnn_layers'],
                                      name_prefix='{0}-fw'.format(TRANSFORM_LAYER_NAME))

                bw_cell = tf_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                      num_units=state_size,
                                      activation=self.hypers.model_params['rnn_activation'],
                                      layers=self.hypers.model_params['rnn_layers'],
                                      name_prefix='{0}-bw'.format(TRANSFORM_LAYER_NAME))

                fw_initial_state = fw_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
                bw_initial_state = bw_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

                transformed, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                     cell_bw=bw_cell,
                                                                     inputs=input_sequence,
                                                                     initial_state_fw=fw_initial_state,
                                                                     initial_state_bw=bw_initial_state,
                                                                     dtype=tf.float32,
                                                                     scope=BIRNN_NAME)
                # Concatenate forward and backward states / outputs
                transformed = tf.concat(transformed, axis=-1)  # [B, T, 2 * D]
            else:
                # Make the RNN cells
                fw_cell = make_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                        input_units=state_size,
                                        output_units=state_size,
                                        activation=self.hypers.model_params['rnn_activation'],
                                        num_layers=self.hypers.model_params['rnn_layers'],
                                        name=TRANSFORM_LAYER_NAME,
                                        compression_fraction=compression_fraction,
                                        compression_seed=TRANSFORM_SEED)

                bw_cell = make_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                        input_units=state_size,
                                        output_units=state_size,
                                        activation=self.hypers.model_params['rnn_activation'],
                                        num_layers=self.hypers.model_params['rnn_layers'],
                                        name=get_backward_name(TRANSFORM_LAYER_NAME),
                                        compression_fraction=compression_fraction,
                                        compression_seed=get_backward_name(TRANSFORM_SEED))

                # Create the initial states
                fw_initial_state = fw_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
                bw_initial_state = bw_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

                # Run RNN in both directions
                fw_output = dynamic_rnn(cell=fw_cell,
                                        inputs=input_sequence,
                                        previous_states=None,
                                        initial_state=fw_initial_state,
                                        name=RNN_NAME,
                                        compression_fraction=compression_fraction)
                bw_output = dynamic_rnn(cell=bw_cell,
                                        inputs=input_sequence,
                                        previous_states=None,
                                        initial_state=fw_initial_state,
                                        name=get_backward_name(RNN_NAME),
                                        compression_fraction=compression_fraction,
                                        should_reverse=True)

                # Get the outputs from each direction
                fw_outputs = fw_output.outputs.stack()  # [T, B, D]
                bw_outputs = bw_output.outputs.stack()  # [T, B, D]
                transformed = tf.concat([fw_outputs, bw_outputs], axis=-1)  # [T, B, 2 * D]
                transformed = tf.transpose(transformed, perm=[1, 0, 2])  # [B, T, 2 * D]
        else:
            raise ValueError(f'Unknown transformation type: {0}'.format(self.model_type))

        # Apply dropout to the aggregated state
        transformed = tf.nn.dropout(transformed, keep_prob=self._placeholders[DROPOUT_KEEP_RATE])

        # Create the output layer, result is a [B, T, C] tensor
        output_size = self.metadata[NUM_OUTPUT_FEATURES] if self.output_type != OutputType.MULTI_CLASSIFICATION else self.metadata[NUM_CLASSES]
        output, _ = mlp(inputs=transformed,
                        output_size=output_size,
                        hidden_sizes=self.hypers.model_params['output_hidden_units'],
                        activations=self.hypers.model_params['output_hidden_activation'],
                        dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                        should_bias_final=True,
                        should_activate_final=False,
                        should_dropout_final=False,
                        name=OUTPUT_LAYER_NAME,
                        compression_fraction=compression_fraction,
                        compression_seed=OUTPUT_SEED)

        # Reshape the output to match the sequence length. The output is tiled along the sequence length
        # automatically via broadcasting rules.
        expected_output = tf.expand_dims(self._placeholders[OUTPUT], axis=-1)  # [B, 1, 1]

        if self.output_type == OutputType.BINARY_CLASSIFICATION:
            classification_output = compute_binary_classification_output(model_output=output,
                                                                         labels=expected_output)

            self._ops[LOGITS] = classification_output.logits
            self._ops[PREDICTION] = classification_output.predictions
            self._ops[ACCURACY] = classification_output.accuracy
            self._ops[F1_SCORE] = classification_output.f1_score
        elif self.output_type == OutputType.MULTI_CLASSIFICATION:
            classification_output = compute_multi_classification_output(model_output=output,
                                                                        labels=expected_output)
            self._ops[LOGITS] = classification_output.logits
            self._ops[PREDICTION] = classification_output.predictions
            self._ops[ACCURACY] = classification_output.accuracy
            self._ops[F1_SCORE] = classification_output.f1_score
        else:
            self._ops[PREDICTION] = output

    def make_loss(self):
        # Tile the output along all sequence elements. This is necessary for the sparse softmax
        # cross entropy function.
        seq_length = self.metadata[SEQ_LENGTH]
        expected_output = tf.expand_dims(self._placeholders[OUTPUT], axis=-1)  # [B, 1, 1]
        expected_output = tf.tile(expected_output, multiples=(1, seq_length, 1))  # [B, T, 1]
        
        predictions = self._ops[PREDICTION]  # [B, T, C]
       
        # Create the loss weights
        if self.hypers.model_params.get('use_loss_weights', False):
            loss_weights = np.linspace(start=(1.0 / seq_length), stop=1.0, endpoint=True, num=seq_length)
        else:
            loss_weights = np.ones(shape=seq_length) / seq_length

        # Expand for later broadcasting
        loss_weights = np.expand_dims(loss_weights, axis=0)  # [1, T]

        if self.output_type == OutputType.BINARY_CLASSIFICATION:
            logits = self._ops[LOGITS]
            sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=expected_output,
                                                                  logits=logits)
        elif self.output_type == OutputType.MULTI_CLASSIFICATION:
            logits = self._ops[LOGITS]
            labels = tf.squeeze(expected_output, axis=-1)  # [B, T]

            sample_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        else:
            sample_loss = tf.square(predictions - expected_output)

        # Compute weighted average over samples and unweighted average over batch
        print(loss_weights)
        print(loss_weights.shape)
        self._ops[LOSS] = tf.reduce_mean(tf.reduce_sum(sample_loss * loss_weights, axis=-1))

        # Add any regularization to the loss function
        reg_loss = self.regularize_weights(name=self.hypers.model_params.get('regularization_name'),
                                           scale=self.hypers.model_params.get('regularization_scale'))
        if reg_loss is not None:
            self._ops[LOSS] += reg_loss

    def compute_flops(self, level: int) -> int:
        """
        Returns the total number of floating point operations to produce the final output.
        """
        total_flops = 0

        with self.sess.graph.as_default():

            # Get FLOPS for operations that are applied to each sequence element
            seq_operations = [TRANSFORM_LAYER_NAME, RNN_NAME, BIRNN_NAME]
            seq_operations = list(map(lambda t: NODE_REGEX_FORMAT.format(t), seq_operations))

            seq_options = tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.float_operation()) \
                                        .with_node_names(show_name_regexes=seq_operations) \
                                        .order_by('flops').build()
            flops = tf.profiler.profile(self.sess.graph, options=seq_options)

            if self.model_type == StandardModelType.NBOW:
                total_flops += flops.total_float_ops
            else:
                total_flops += flops.total_float_ops * self.metadata[SEQ_LENGTH]

            # Get FLOPS for operations that are applied to the entire sequence. We include the embedding layer
            # here because it has a well-defined sequence length so Tensorflow will automatically account for
            # the multiplier
            single_operations = list(map(lambda t: NODE_REGEX_FORMAT.format(t), [OUTPUT_LAYER_NAME, EMBEDDING_LAYER_NAME, AGGREGATION_LAYER_NAME]))
            single_options = tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.float_operation()) \
                                            .with_node_names(show_name_regexes=single_operations) \
                                            .order_by('flops').build()
            flops = tf.profiler.profile(self.sess.graph, options=single_options)
            total_flops += flops.total_float_ops

        return total_flops

    def predict_classification(self, test_batch_generator: Iterable[Any],
                               batch_size: int,
                               max_num_batches: Optional[int],
                               flops_dict: Dict[str, int]) -> DefaultDict[str, Dict[str, Any]]:
        predictions_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        latencies: List[float] = []

        for batch_num, batch in enumerate(test_batch_generator):
            feed_dict = self.batch_to_feed_dict(batch, is_train=False)

            start = time.time()
            prediction = self.sess.run(self._ops[PREDICTION], feed_dict=feed_dict)
            elapsed = time.time() - start

            labels_list.append(np.vstack(batch[OUTPUT]))
            predictions_list.append(np.vstack(prediction))
            latencies.append(elapsed)

        predictions = np.vstack(predictions_list)  # [B, T]
        labels = np.squeeze(np.vstack(labels_list), axis=-1)  # [B]

        avg_latency = np.average(latencies[1:])  # Skip first due to outliers in caching
        flops = flops_dict[self.output_ops[0]]

        result: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
        for i in range(self.metadata[SEQ_LENGTH]):
            level_name = '{0}_{1}'.format(PREDICTION, i)

            for metric_name in ClassificationMetric:
                if self.output_type == OutputType.BINARY_CLASSIFICATION:
                    metric_value = get_binary_classification_metric(metric_name, predictions[:, i], labels, avg_latency, 1, flops)
                else:
                    metric_value = get_multi_classification_metric(metric_name, predictions[:, i], labels, avg_latency, 1, flops, self.metadata[NUM_CLASSES])

                result[level_name][metric_name.name] = metric_value

            result[level_name][ALL_LATENCY] = latencies[1:]

        return result
