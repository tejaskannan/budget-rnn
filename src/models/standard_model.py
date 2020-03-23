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
from utils.constants import ACCURACY, ONE_HALF, OUTPUT, INPUTS, LOSS, PREDICTION, F1_SCORE, LOGITS
from utils.testing_utils import ClassificationMetric, RegressionMetric, get_classification_metric, get_regression_metric
from .base_model import Model


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

        self.metadata['input_scaler'] = input_scaler
        self.metadata['output_scaler'] = output_scaler
        self.metadata['input_shape'] = input_shape
        self.metadata['num_output_features'] = num_output_features
        self.metadata['seq_length'] = seq_length

    def batch_to_feed_dict(self, batch: Dict[str, List[Any]], is_train: bool) -> Dict[tf.Tensor, np.ndarray]:
        dropout = self.hypers.dropout_keep_rate if is_train else 1.0
        input_batch = np.array(batch[INPUTS])
        output_batch = np.array(batch[OUTPUT])

        if input_batch.shape[1] == 1:
            input_batch = np.squeeze(input_batch, axis=1)

        input_shape = self.metadata['input_shape']
        num_output_features = self.metadata['num_output_features']

        feed_dict = {
            self._placeholders[INPUTS]: input_batch,
            self._placeholders[OUTPUT]: output_batch.reshape(-1, num_output_features),
            self._placeholders['dropout_keep_rate']: dropout
        }

        return feed_dict

    def make_placeholders(self):
        input_features_shape = self.metadata['input_shape']
        num_output_features = self.metadata['num_output_features']
        seq_length = self.metadata['seq_length']

        input_shape = (None, seq_length) + input_features_shape
        self._placeholders[INPUTS] = tf.placeholder(shape=input_shape,
                                                    dtype=tf.float32,
                                                    name=INPUTS)

        self._placeholders[OUTPUT] = tf.placeholder(shape=[None, num_output_features],
                                                    dtype=tf.float32,
                                                    name=OUTPUT)
        self._placeholders['dropout_keep_rate'] = tf.placeholder(shape=[],
                                                                 dtype=tf.float32,
                                                                 name='dropout-keep-rate')

    def make_model(self, is_train: bool):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self._make_model(is_train)

    def _make_model(self, is_train: bool):
    
        # Embed the input sequence into a [B, T, D] tensor
        input_sequence = embedding_layer(inputs=self._placeholders[INPUTS],
                                         units=self.hypers.model_params['state_size'],
                                         dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                                         use_conv=self.hypers.model_params['use_conv_embedding'],
                                         params=self.hypers.model_params['embedding_layer_params'],
                                         seq_length=self.metadata['seq_length'],
                                         input_shape=self.metadata['input_shape'],
                                         name_prefix=f'embedding-layer')

        # Apply the transformation layer
        if self.model_type == StandardModelType.NBOW:
            # Apply the MLP transformation. Outputs a [B, T, D] tensor
            transformed = mlp(inputs=input_sequence,
                              output_size=self.hypers.model_params['state_size'],
                              hidden_sizes=self.hypers.model_params['mlp_hidden_units'],
                              activations=self.hypers.model_params['mlp_activation'],
                              dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                              should_activate_final=True,
                              should_bias_final=True,
                              should_dropout_final=True,
                              name='transform-layer')

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
                                           name='transform-layer')

            aggregated = pool_sequence(transformed, pool_mode=self.hypers.model_params['pool_mode'])
        elif self.model_type == StandardModelType.RNN:
            cell = tf_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                               num_units=self.hypers.model_params['state_size'],
                               activation=self.hypers.model_params['rnn_activation'],
                               layers=self.hypers.model_params['rnn_layers'],
                               dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                               name_prefix='transform-layer')

            initial_state = cell.zero_state(batch_size=tf.shape(input_sequence)[0], dtype=tf.float32)
            transformed, state = tf.nn.dynamic_rnn(cell=cell,
                                                   inputs=input_sequence,
                                                   initial_state=initial_state,
                                                   dtype=tf.float32,
                                                   scope='rnn')
            final_state = get_rnn_state(state)

            aggregated = pool_rnn_outputs(transformed, final_state, pool_mode=self.hypers.model_params['pool_mode'])
        elif self.model_type == StandardModelType.BIRNN:
            fw_cell = tf_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                  num_units=self.hypers.model_params['state_size'],
                                  activation=self.hypers.model_params['rnn_activation'],
                                  layers=self.hypers.model_params['rnn_layers'],
                                  dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                                  name_prefix='transform-layer-fw')
            
            bw_cell = tf_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                  num_units=self.hypers.model_params['state_size'],
                                  activation=self.hypers.model_params['rnn_activation'],
                                  layers=self.hypers.model_params['rnn_layers'],
                                  dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                                  name_prefix='transform-layer-bw')

            batch_size = tf.shape(input_sequence)[0]
            fw_initial_state = fw_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            bw_initial_state = bw_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            transformed, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                 cell_bw=bw_cell,
                                                                 inputs=input_sequence,
                                                                 initial_state_fw=fw_initial_state,
                                                                 initial_state_bw=bw_initial_state,
                                                                 dtype=tf.float32,
                                                                 scope='birnn')
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
                     output_size=self.metadata['num_output_features'],
                     hidden_sizes=self.hypers.model_params['output_hidden_units'],
                     activations=self.hypers.model_params['output_hidden_activation'],
                     dropout_keep_rate=self._placeholders['dropout_keep_rate'],
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

    def predict_classification(self, test_batch_generator: Iterable[Any],
                               batch_size: int,
                               max_num_batches: Optional[int]) -> DefaultDict[str, Dict[str, Any]]:
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

        result: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
        for metric_name in ClassificationMetric:
            metric_value = get_classification_metric(metric_name, predictions, labels, avg_latency, 1)
            result['model'][metric_name.name] = metric_value

        result['model']['ALL_LATENCY'] = latencies[1:]

        return result
