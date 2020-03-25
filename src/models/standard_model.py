import tensorflow as tf
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from enum import Enum, auto
from collections import defaultdict
from typing import Optional, Dict, List, Any, DefaultDict, Iterable

from layers.basic import mlp, pool_sequence
from layers.output_layers import OutputType, compute_binary_classification_output, compute_regression_output
from layers.embedding_layer import embedding_layer
from dataset.dataset import Dataset, DataSeries
from utils.hyperparameters import HyperParameters
from utils.tfutils import pool_rnn_outputs, get_activation, tf_rnn_cell, get_rnn_state
from utils.constants import ACCURACY, ONE_HALF, OUTPUT, INPUTS, LOSS, PREDICTION, F1_SCORE, LOGITS, NODE_REGEX_FORMAT
from utils.constants import INPUT_SHAPE, NUM_OUTPUT_FEATURES, INPUT_SCALER, OUTPUT_SCALER, SEQ_LENGTH, DROPOUT_KEEP_RATE, MODEL
from utils.testing_utils import ClassificationMetric, RegressionMetric, get_classification_metric, get_regression_metric, ALL_LATENCY
from .base_model import Model


# Layer name constants
EMBEDDING_LAYER_NAME = 'embedding-layer'
TRANSFORM_LAYER_NAME = 'transform-layer'
OUTPUT_LAYER_NAME = 'output-layer'
RNN_NAME = 'rnn'
BIRNN_NAME = 'birnn'


class StandardModelType(Enum):
    NBOW = auto()
    CNN = auto()
    RNN = auto()
    BIRNN = auto()


class StandardModel(Model):

    def __init__(self, hyper_parameters: HyperParameters, save_folder: str):
        super().__init__(hyper_parameters, save_folder)

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

    def load_metadata(self, dataset: Dataset):
        input_samples: List[List[float]] = []
        output_samples: List[List[float]] = []

        # Fetch training samples to prepare for normalization
        for sample in dataset.iterate_series(series=DataSeries.TRAIN):
            input_sample = np.array(sample[INPUTS])
            input_samples.append(input_sample)

            if not isinstance(sample[OUTPUT], list) and \
                    not isinstance(sample[OUTPUT], np.ndarray):
                output_samples.append([sample[OUTPUT]])
            elif isinstance(sample[OUTPUT], np.ndarray) and len(sample[OUTPUT].shape) == 0:
                output_samples.append([sample[OUTPUT]])
            else:
                output_samples.append(sample[OUTPUT])

        # Infer the number of input and output features
        first_sample = np.array(input_samples[0])
        input_shape = first_sample.shape[1:]  # Skip the sequence length
        seq_length = len(input_samples[0])

        input_scaler = None
        if self.hypers.model_params['normalize_inputs']:
            assert len(input_shape) == 1
            input_samples = np.reshape(input_samples, newshape=(-1, input_shape[0]))
            input_scaler = StandardScaler()
            input_scaler.fit(input_samples)

        output_scaler = None
        num_output_features = len(output_samples[0])
        if self.output_type == OutputType.REGRESSION:
            output_scaler = StandardScaler()
            output_scaler.fit(output_samples)

        self.metadata[INPUT_SCALER] = input_scaler
        self.metadata[OUTPUT_SCALER] = output_scaler
        self.metadata[INPUT_SHAPE] = input_shape
        self.metadata[NUM_OUTPUT_FEATURES] = num_output_features
        self.metadata[SEQ_LENGTH] = seq_length

    def batch_to_feed_dict(self, batch: Dict[str, List[Any]], is_train: bool) -> Dict[tf.Tensor, np.ndarray]:
        dropout = self.hypers.dropout_keep_rate if is_train else 1.0
        input_batch = np.array(batch[INPUTS])
        output_batch = np.array(batch[OUTPUT])

        if input_batch.shape[1] == 1:
            input_batch = np.squeeze(input_batch, axis=1)

        input_shape = self.metadata[INPUT_SHAPE]
        num_output_features = self.metadata[NUM_OUTPUT_FEATURES]

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

        if not is_frozen:
            self._placeholders[INPUTS] = tf.placeholder(shape=input_shape,
                                                        dtype=tf.float32,
                                                        name=INPUTS)

            self._placeholders[OUTPUT] = tf.placeholder(shape=(None, num_output_features),
                                                        dtype=tf.float32,
                                                        name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = tf.placeholder(shape=(),
                                                                   dtype=tf.float32,
                                                                   name=DROPOUT_KEEP_RATE)
        else:
            self._placeholders[INPUTS] = tf.ones(shape=(1,) + input_shape[1:], dtype=tf.float32, name=INPUTS)
            self._placeholders[OUTPUT] = tf.ones(shape=(1, num_output_features), dtype=tf.float32, name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = tf.ones(shape=(), dtype=tf.float32, name=DROPOUT_KEEP_RATE)

    def make_model(self, is_train: bool):
        with tf.variable_scope(MODEL, reuse=tf.AUTO_REUSE):
            self._make_model(is_train)

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

            aggregated = pool_sequence(transformed, pool_mode=self.hypers.model_params['pool_mode'])
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
            # Apply dropout
            transformed = tf.nn.dropout(transformed, keep_rate=self._placeholders[DROPOUT_KEEP_RATE])

            aggregated = pool_sequence(transformed, pool_mode=self.hypers.model_params['pool_mode'])
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
            final_state = get_rnn_state(state)

            aggregated = pool_rnn_outputs(transformed, final_state, pool_mode=self.hypers.model_params['pool_mode'])
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
            # Concatenate forward and backward states
            transformed = tf.concat(transformed, axis=-1)

            fw_state, bw_state = state
            final_state_fw = get_rnn_state(fw_state)
            final_state_bw = get_rnn_state(bw_state)
            final_state = tf.concat([final_state_fw, final_state_bw], axis=-1)

            aggregated = pool_rnn_outputs(transformed, final_state, pool_mode=self.hypers.model_params['pool_mode'])
        else:
            raise ValueError(f'Unknown transformation type: {self.model_type}')

        # Create the output layer
        output = mlp(inputs=aggregated,
                     output_size=self.metadata[NUM_OUTPUT_FEATURES],
                     hidden_sizes=self.hypers.model_params['output_hidden_units'],
                     activations=self.hypers.model_params['output_hidden_activation'],
                     dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                     name='output-layer')

        if self.output_type == OutputType.CLASSIFICATION:
            classification_output = compute_binary_classification_output(model_output=output,
                                                                         labels=self._placeholders[OUTPUT],
                                                                         false_pos_weight=1.0,
                                                                         false_neg_weight=1.0,
                                                                         mode=self.hypers.model_params['loss_mode'])

            self._ops[LOGITS] = classification_output.logits
            self._ops[PREDICTION] = classification_output.predictions
            self._ops[LOSS] = classification_output.loss
            self._ops[ACCURACY] = classification_output.accuracy
            self._ops[F1_SCORE] = classification_output.f1_score
        else:
            regression_output = compute_regression_output(model_output=output, expected_otuput=self._placeholders[OUTPUT])
            self._ops[PREDICTION] = regression_output.predictions
            self._ops[LOSS] = regression_output.loss

    def make_loss(self):
        pass  # Loss made during model creation

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
            single_operations = list(map(lambda t: NODE_REGEX_FORMAT.format(t), [OUTPUT_LAYER_NAME, TRANSFORM_LAYER_NAME]))
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
            predictions_list.append(prediction)
            latencies.append(elapsed)

        predictions = np.vstack(predictions_list)
        labels = np.vstack(labels_list)
        avg_latency = np.average(latencies[1:])  # Skip first due to outliers in caching
        flops = flops_dict[self.output_ops[0]]

        result: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
        for metric_name in ClassificationMetric:
            metric_value = get_classification_metric(metric_name, predictions, labels, avg_latency, 1, flops)
            result[MODEL][metric_name.name] = metric_value

        result[MODEL][ALL_LATENCY] = latencies[1:]

        return result
