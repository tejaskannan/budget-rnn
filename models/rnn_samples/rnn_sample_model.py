import tensorflow as tf

import numpy as np
import re
import time
from enum import Enum, auto
from collections import namedtuple, defaultdict, OrderedDict
from typing import List, Optional, Tuple, Dict, Any, Set, Union, DefaultDict, Iterable
from sklearn.preprocessing import StandardScaler
from dpu_utils.utils import RichPath

from models.base_model import Model, VariableWithWeight
from layers.basic import rnn_cell, mlp
from layers.cells.cells import make_rnn_cell, MultiRNNCell
from layers.rnn import dynamic_rnn, dropped_rnn
from dataset.dataset import Dataset, DataSeries
from utils.hyperparameters import HyperParameters
from utils.tfutils import pool_rnn_outputs
from utils.constants import SMALL_NUMBER
from rnn_samples.testing_utils import TestMetrics, Prediction


VAR_NAME_REGEX = re.compile(r'.*(layer-[0-9])+.*')


class RNNModelType:
    VANILLA = auto()
    SAMPLE = auto()
    CASCADE = auto()
    CHUNKED = auto()
    SKIP = auto()
    DROPPED = auto()
    SINGLE = auto()


def var_filter(trainable_vars: List[tf.Variable], var_prefixes: Optional[Dict[str, float]] = None) -> List[VariableWithWeight]:
    result: List[VariableWithWeight] = []
    for var in trainable_vars:
        name_match = VAR_NAME_REGEX.match(var.name)
        layer_name = name_match.group(1) if name_match is not None else None
        if var_prefixes is None or layer_name in var_prefixes:
            weight = 1.0 if var_prefixes is None else var_prefixes[layer_name]
            result.append(VariableWithWeight(var, weight))
    return result


class RNNSampleModel(Model):

    def __init__(self, hyper_parameters: HyperParameters, save_folder: Union[str, RichPath]):
        super().__init__(hyper_parameters, save_folder)

        model_type = self.hypers.model_params.get('model_type', '').lower()
        if model_type == 'vanilla_rnn_model':
            self.model_type = RNNModelType.VANILLA
        elif model_type == 'sample_rnn_model':
            self.model_type = RNNModelType.SAMPLE
        elif model_type == 'cascade_rnn_model':
            self.model_type = RNNModelType.CASCADE
        elif model_type == 'chunked_rnn_model':
            assert not self.hypers.model_params['share_cell_weights'], 'The chunked model cannot share cell weights.'
            self.model_type = RNNModelType.CHUNKED
        elif model_type == 'skip_rnn_model':
            self.model_type = RNNModelType.SKIP
        elif model_type == 'dropped_rnn_model':
            self.model_type = RNNModelType.DROPPED
        elif model_type == 'single_rnn_model':
            self.model_type = RNNModelType.SINGLE
        else:
            raise ValueError(f'Unknown model type: {model_type}.')

        self.name = model_type

    @property
    def num_sequences(self) -> int:
        return int(1.0 / self.hypers.model_params['sample_frac'])

    @property
    def num_outputs(self) -> int:
        return int(1.0 / self.hypers.model_params['sample_frac'])

    @property
    def prediction_ops(self) -> List[str]:
        return [f'prediction_{i}' for i in range(self.num_outputs)]

    @property
    def accuracy_op_names(self) -> List[str]:
        return [f'accuracy_{i}' for i in range(self.num_outputs)]

    @property
    def samples_per_seq(self) -> int:
        seq_length = self.metadata['seq_length']
        return int(seq_length * self.hypers.model_params['sample_frac'])

    @property
    def loss_op_names(self) -> List[str]:
        if self.model_type == RNNModelType.VANILLA:
            return [f'loss_{i}' for i in range(self.num_outputs)]
        return ['loss']

    def get_variable_group(self, loss_op_name: str) -> List[tf.Variable]:
        variables: List[tf.Variable] = []
        trainable_vars = list(self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        if self.model_type != RNNModelType.VANILLA:
            return trainable_vars

        if self.hypers.model_params.get('share_cell_weights', False):
            variables.extend(filter(lambda v: 'rnn-cell' in v.name or 'rnn-model' in v.name, trainable_vars))

        loss_index = int(loss_op_name[-1])
        variables.extend(filter(lambda v: f'layer-{loss_index}' in v.name, trainable_vars))
        return variables

    def load_metadata(self, dataset: Dataset):
        input_samples: List[List[float]] = []
        output_samples: List[List[float]] = []

        for sample in dataset.dataset[DataSeries.TRAIN]:

            # Shift input by the first sample to focus on trends
            input_sample = np.array(sample['inputs'])
            if self.hypers.model_params.get('shift_inputs', False):
                first_input = np.expand_dims(input_sample[0], axis=0)
                shifted_input = input_sample - first_input
                input_samples.append(shifted_input)
            else:
                input_samples.append(input_sample)

            if not isinstance(sample['output'], list) and \
                    not isinstance(sample['output'], np.ndarray):
                output_samples.append([sample['output']])
            else:
                output_samples.append(sample['output'])

        num_input_features = len(input_samples[0][0])
        seq_length = len(input_samples[0])
        input_samples = np.reshape(input_samples, newshape=(-1, num_input_features))

        input_scaler = StandardScaler()
        input_scaler.fit(input_samples)

        num_output_features = len(output_samples[0])
        output_scaler = StandardScaler()
        output_scaler.fit(output_samples)

        if self.hypers.model_params.get('bin_outputs', False):
            assert num_output_features == 1, 'Can only bin when the number of output features is one'

            normalized_outputs = output_scaler.transform(output_samples)
            sorted_outputs = np.sort(normalized_outputs, axis=0)

            bin_bounds: List[Tuple[float, float]] = []
            bin_means: List[float] = []
            stride = int(sorted_outputs.shape[0] / (self.hypers.model_params['num_bins'] - 1))
            for i in range(0, sorted_outputs.shape[0], stride):
                split = sorted_outputs[i:i+stride, :]
                bin_bouds.append((np.min(split), np.max(split)))
                bin_meas.append(np.average(split))

            self.metadaa['bin_bounds'] = bin_bounds
            self.metadaa['bin_means'] = bin_means

        self.metadata['input_scaler'] = input_scaler
        self.metadata['output_scaler'] = output_scaler
        self.metadata['num_input_features'] = num_input_features
        self.metadata['num_output_features'] = num_output_features
        self.metadata['seq_length'] = seq_length
        self.metadata['shift_inputs'] = self.hypers.model_params.get('shift_inputs', False)
        self.metadata['normalize_output'] = self.hypers.model_params['output_type'].lower() != 'classification'

    def batch_to_feed_dict(self, batch: Dict[str, List[Any]], is_train: bool) -> Dict[tf.Tensor, np.ndarray]:
        dropout = self.hypers.dropout_keep_rate if is_train else 1.0
        input_batch = np.array(batch['inputs'])
        output_batch = np.array(batch['output'])

        input_batch = np.squeeze(input_batch, axis=1)

        num_input_features = self.metadata['num_input_features']
        num_output_features = self.metadata['num_output_features']

        feed_dict = {
            self._placeholders['output']: output_batch.reshape(-1, num_output_features),
            self._placeholders['dropout_keep_rate']: dropout
        }

        if self.hypers.model_params.get('bin_outputs', False):
            feed_dict[self._placeholders['bin_means']] = batch['bin_means']

        # Extract parameters
        seq_length = self.metadata['seq_length']
        samples_per_seq = self.samples_per_seq
        num_sequences = self.num_sequences
        num_outputs = self.num_outputs

        loss_weights = self.hypers.model_params.get('loss_weights')
        if loss_weights is None:
            loss_weights = np.ones(shape=num_outputs, dtype=float)
        elif isinstance(loss_weights, float):
            loss_weights = [loss_weights] * num_outputs
        elif len(loss_weights) == 1:
            loss_weights = loss_weight * num_outputs

        assert len(loss_weights) == num_outputs, f'Loss weights ({len(loss_weights)}) must match the number of outputs ({num_outputs}).'

        # The loss weights are sorted in ascending order to more-heavily weight larger sample sizes.
        # This operation is included to prevent bugs as we intuitively want to increase
        # accuracy with larger samples.
        loss_weights = list(sorted(loss_weights))
        feed_dict[self._placeholders['loss_weights']] = loss_weights / (np.sum(loss_weights) + 1e-7)  # Normalize the loss weights

        seq_indexes: List[int] = []
        for i in range(num_sequences):
            if self.model_type in (RNNModelType.VANILLA, RNNModelType.DROPPED, RNNModelType.SINGLE):
                seq_indexes.extend(range(i, seq_length, num_sequences))
                seq_indexes = list(sorted(seq_indexes))
                sample_tensor = input_batch[:, seq_indexes]
                feed_dict[self._placeholders[f'input_{i}']] = sample_tensor
            elif self.model_type == RNNModelType.SAMPLE:
                seq_indexes = list(range(i, seq_length, num_sequences))
                sample_tensor = input_batch[:, seq_indexes]
                feed_dict[self._placeholders[f'input_{i}']] = sample_tensor
            else:  # Chunked or Cascade
                start, end = i * samples_per_seq, (i+1) * samples_per_seq
                sample_tensor = input_batch[:, start:end]
                feed_dict[self._placeholders[f'input_{i}']] = sample_tensor

        return feed_dict

    def make_placeholders(self):
        """
        Create model placeholders.
        """
        # Extract parameters
        num_input_features = self.metadata['num_input_features']
        num_output_features = self.metadata['num_output_features']
        seq_length = self.metadata['seq_length']
        samples_per_seq = self.samples_per_seq
        num_sequences = self.num_sequences

        # Make input placeholders
        for i in range(num_sequences):
            input_shape = [None, samples_per_seq, num_input_features]

            # B x S x D
            self._placeholders[f'input_{i}'] = tf.placeholder(shape=input_shape,
                                                              dtype=tf.float32,
                                                              name=f'input-{i}')
            if self.model_type in (RNNModelType.VANILLA, RNNModelType.DROPPED, RNNModelType.SINGLE):
                samples_per_seq += self.samples_per_seq

        self._placeholders['output'] = tf.placeholder(shape=[None, num_output_features],
                                                      dtype=tf.float32,
                                                      name='output')  # B x K
        self._placeholders['dropout_keep_rate'] = tf.placeholder(shape=[], dtype=tf.float32, name='dropout-keep-rate')
        self._placeholders['loss_weights'] = tf.placeholder(shape=[self.num_outputs],
                                                            dtype=tf.float32,
                                                            name='loss-weights')

        if self.hypers.model_params.get('bin_outputs', False):
            self._placeholders['bin_means'] = tf.placeholder(shape=[None, self.hypers.model_params['num_bins']],
                                                             dtype=tf.float32,
                                                             name='bin-means')

    def predict(self, dataset: Dataset, name: str,
                test_batch_size: Optional[int] = None,
                max_num_batches: Optional[int] = None) -> TestMetrics:
        test_batch_size = test_batch_size if test_batch_size is not None else self.hypers.batch_size
        test_batch_generator = dataset.minibatch_generator(series=DataSeries.TEST,
                                                           batch_size=test_batch_size,
                                                           metadata=self.metadata,
                                                           should_shuffle=False,
                                                           drop_incomplete_batches=True)
        prediction_ops = self.prediction_ops

        sq_error_dict: DefaultDict[str, List[float]] = defaultdict(list)
        abs_error_dict: DefaultDict[str, List[float]] = defaultdict(list)
        abs_perc_dict: DefaultDict[str, List[float]] = defaultdict(list)
        latency_dict: DefaultDict[str, List[float]] = defaultdict(list)
        gate_dict: DefaultDict[str, Dict[str, List[float]]] = defaultdict(dict)
        predictions_dict: DefaultDict[str, List[Prediction]] = defaultdict(list)

        true_values: List[np.array] = []

        num_batches = 0
        for batch in test_batch_generator:
            feed_dict = self.batch_to_feed_dict(batch, is_train=False)
            op_results: Dict[str, Any] = dict()

            # Execute operations individually for better profiling
            for i in range(len(prediction_ops)):
                ops_to_run = prediction_ops[0:i+1]
                ops_to_run.append(f'gates_{i}')
                ops_to_run.append(f'states_{i}')

                start = time.time()
                result = self.execute(feed_dict, ops_to_run)
                elapsed = time.time() - start

                # Save results
                op = prediction_ops[i]
                op_results[op] = result[op]

                # Do not accumulate latency metrics on first batch (to avoid outliers from caching)
                if num_batches == 0:
                    continue

                latency_dict[op].append(elapsed * 1000.0)  # Latency in seconds

                # The gates depend on the RNN type
                #if self.hypers.model_params['rnn_cell_type'] == 'gru':
                #    gates: DefaultDict[str, List[float]] = defaultdict(list)
                #    gate_values = result[f'gates_{i}']

                #    gates['update'].extend(gate_values[:, :, 0, :].reshape(-1))
                #    gates['reset'].extend(gate_values[:, :, 1, :].reshape(-1))

                #    gate_dict[op] = gates

            # Avoid analyzing the predictions for the first batch for consistency with the latency
            # measurements
            if num_batches == 0:
                num_batches += 1
                continue

            for prediction_op in prediction_ops:
                prediction = op_results[prediction_op]

                raw_expected = np.array(batch['output']).reshape(-1, self.metadata['num_output_features'])

                if self.hypers.model_params['output_type'].lower() == 'classification':
                    unnormalized_prediction = prediction
                    expected = raw_expected
                else:
                    unnormalized_prediction = self.metadata['output_scaler'].inverse_transform(prediction)
                    raw_expected = np.array(batch['output']).reshape(-1, self.metadata['num_output_features'])
                    expected = self.metadata['output_scaler'].inverse_transform(raw_expected)

                true_values.append(expected)

                difference = expected - unnormalized_prediction
                squared_error = np.sum(np.square(difference), axis=-1)
                abs_error = np.sum(np.abs(difference), axis=-1)
                abs_percentage_error = np.sum(np.abs(difference) / (expected + 1e-7), axis=-1)

                predictions = [Prediction(sample_id=s, prediction=p[0], expected=e[0]) for s, p, e in zip(batch['sample_id'], unnormalized_prediction, expected)]

                sq_error_dict[prediction_op].extend(squared_error)
                abs_error_dict[prediction_op].extend(abs_error)
                abs_perc_dict[prediction_op].extend(abs_percentage_error)
                predictions_dict[prediction_op].extend(predictions)

            num_batches += 1

            if max_num_batches is not None and num_batches >= max_num_batches:
                break

        return TestMetrics(squared_error=sq_error_dict,
                           abs_error=abs_error_dict,
                           abs_percentage_error=abs_perc_dict,
                           latency=latency_dict,
                           gate_values=gate_dict,
                           predictions=predictions_dict)

    def make_model(self, is_train: bool):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            if self.model_type == RNNModelType.SAMPLE:
                self.make_rnn_sample_model(is_train)
            else:
                self.make_rnn_model(is_train)

    def classification_loss(self, predicted_probs: tf.Tensor, predictions: tf.Tensor, labels: tf.Tensor, pos_weight: float, neg_weight: float):
        # False negative -> predict 0 when label is actually 1
        # False positive -> predict 1 when label is actually 0
        # If prediction = 0 and label = 1 -> false negative
        # If prediction = 1 and label = 0 -> false positive
        # If prediction = 0 and label = 0 -> (1 - 

        log_probs = -tf.log(predicted_probs)
        log_opp_probs = -tf.log(1.0 - predicted_probs)
        return tf.where((1.0 - tf.abs(predictions - labels)) < SMALL_NUMBER,
                        x=neg_weight * labels * log_probs + pos_weight * (1.0 - labels) * log_opp_probs,  # False negative or False Positive
                        y=labels * log_probs + (1.0 - labels) * log_opp_probs)

    def make_rnn_model(self, is_train: bool):
        rnn_layers = self.hypers.model_params['rnn_layers']
        outputs: List[tf.Tensor] = []
        prev_state: Optional[tf.Tensor] = None

        for i in range(self.num_sequences):
            rnn_cell_name = 'rnn-cell'
            output_layer_name = 'output-layer'
            rnn_model_name = 'rnn-model'

            if self.model_type in (RNNModelType.CASCADE, RNNModelType.SINGLE):
                # The Cascade model shares cell weights by design
                output_layer_name = f'{output_layer_name}-layer-{i}'
            elif not self.hypers.model_params.get('share_cell_weights', False):
                rnn_cell_name = f'{rnn_cell_name}-layer-{i}'
                output_layer_name = f'{output_layer_name}-layer-{i}'
                rnn_model_name = f'{rnn_model_name}-layer-{i}'

            cell = make_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                 input_units=self.metadata['num_input_features'],
                                 output_units=self.hypers.model_params['state_size'],
                                 activation=self.hypers.model_params['rnn_activation'],
                                 dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                                 num_layers=rnn_layers,
                                 name=rnn_cell_name,
                                 use_skip_connections=self.model_type == RNNModelType.SKIP)

            inputs = self._placeholders[f'input_{i}']
            initial_state = cell.zero_state(batch_size=tf.shape(inputs)[0], dtype=tf.float32)

            if self.model_type in (RNNModelType.CHUNKED, RNNModelType.CASCADE) and prev_state is not None:
                initial_state = prev_state

            if self.model_type == RNNModelType.DROPPED:
                rnn_outputs, rnn_states, rnn_gates = dropped_rnn(cell=cell,
                                                                 inputs=inputs,
                                                                 initial_state=initial_state,
                                                                 horizon=5,
                                                                 drop_rate=0.9,
                                                                 name=rnn_model_name)
            else:
                skip_width = self.num_sequences if self.model_type == RNNModelType.SKIP else None
                rnn_outputs, rnn_states, rnn_gates = dynamic_rnn(cell=cell,
                                                                 inputs=inputs,
                                                                 initial_state=initial_state,
                                                                 skip_width=skip_width,
                                                                 name=rnn_model_name)

            # Get the final state
            last_index = tf.shape(inputs)[1] - 1
            output = rnn_outputs.read(index=last_index)

            # Save previous state for the chunked model
            if self.model_type in (RNNModelType.CHUNKED, RNNModelType.CASCADE):
                prev_state = rnn_states.read(index=last_index)

            # B x D
            rnn_output = pool_rnn_outputs(rnn_outputs, output, pool_mode=self.hypers.model_params['pool_mode'])

            if self.hypers.model_params.get('bin_outputs', False):
                num_output_features = len(self.metadata['bin_means'])
            else:
                num_output_features = self.metadata['num_output_features']

            # B x D'
            output = mlp(inputs=rnn_output,
                         output_size=num_output_features,
                         hidden_sizes=self.hypers.model_params.get('output_hidden_units'),
                         activations=self.hypers.model_params['output_hidden_activation'],
                         dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                         name=output_layer_name)

            if self.hypers.model_params.get('bin_outputs', False):
                output_probs = tf.nn.softmax(output, axis=-1)
                output = tf.reduce_sum(output_probs * self._placeholders['bin_means'], axis=-1, keepdims=True)
                self._ops[f'prediction_probs_{i}'] = output_probs

            if self.hypers.model_params['output_type'].lower() == 'classification':
                self._ops[f'logits_{i}'] = output

                predicted_probs = tf.math.sigmoid(output)
                predictions = tf.cast(predicted_probs > 0.5, tf.float32)
                self._ops[f'prediction_{i}'] = predictions

                self._ops[f'loss_{i}'] = self.classification_loss(predicted_probs=predicted_probs,
                                                                  predictions=predictions,
                                                                  labels=self._placeholders['output'],
                                                                  pos_weight=self.hypers.model_params['pos_weights'][i],
                                                                  neg_weight=self.hypers.model_params['neg_weights'][i])

                self._ops[self.accuracy_op_names[i]] = tf.reduce_mean(1.0 - tf.abs(predictions - self._placeholders['output']))
                output = predictions
            else:
                self._ops[f'prediction_{i}'] = output
                self._ops[f'loss_{i}'] = tf.reduce_sum(tf.square(output - self._placeholders['output']), axis=-1)  # B

            self._ops[f'gates_{i}'] = rnn_gates.stack()  # B x T x M x D
            self._ops[f'states_{i}'] = rnn_states.stack()  # B x T x D

            outputs.append(output)

        self._ops['predictions'] = tf.concat([tf.expand_dims(t, axis=1) for t in outputs], axis=1)

        use_previous_layers = self.model_type == RNNModelType.CHUNKED
        self._loss_ops = self._make_loss_ops(use_previous_layers=use_previous_layers)

    def make_rnn_sample_model(self, is_train: bool):
        outputs: List[tf.Tensor] = []
        states_list: List[tf.TensorArray] = []
        num_sequences = int(1.0 / self.hypers.model_params['sample_frac'])
        rnn_layers = self.hypers.model_params['rnn_layers']

        for i in range(num_sequences):
            # Create the RNN Cell
            cell_name = 'rnn-cell'
            if not self.hypers.model_params['share_cell_weights']:
                cell_name = f'{cell_name}-layer-{i}'  # If not weight sharing, then each cell has its own scope

            cell = make_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                 input_units=self.metadata['num_input_features'],
                                 output_units=self.hypers.model_params['state_size'],
                                 activation=self.hypers.model_params['rnn_activation'],
                                 dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                                 num_layers=rnn_layers,
                                 name=cell_name)

            prev_states = states_list[i-1] if i > 0 else None
            rnn_outputs, states, gates = dynamic_rnn(inputs=self._placeholders[f'input_{i}'],
                                                     cell=cell,
                                                     previous_states=prev_states,
                                                     name=f'rnn-layer-{i}')

            last_index = tf.shape(self._placeholders[f'input_{i}'])[1] - 1
            final_state = states.read(last_index)
            states_list.append(states)

            # B x D
            final_output = rnn_outputs.read(last_index)
            rnn_output = pool_rnn_outputs(rnn_outputs, final_output, pool_mode=self.hypers.model_params['pool_mode'])

            if self.hypers.model_params.get('bin_outputs', False):
                num_output_features = self.hypers.model_params['num_bins']
            else:
                num_output_features = self.metadata['num_output_features']

            # B x D'
            output = mlp(inputs=rnn_output,
                         output_size=num_output_features,
                         hidden_sizes=self.hypers.model_params.get('output_hidden_units'),
                         activations=self.hypers.model_params['output_hidden_activation'],
                         dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                         name=f'output-layer-{i}')

            if self.hypers.model_params.get('bin_outputs', False):
                output_probs = tf.nn.softmax(output, axis=-1)
                output = tf.reduce_sum(output_probs * self._placeholders['bin_means'], axis=-1, keepdims=True)
                self._ops[f'prediction_probs_{i}'] = output_probs

            if self.hypers.model_params['output_type'].lower() == 'classification':
                self._ops[f'logits_{i}'] = output

                predicted_probs = tf.math.sigmoid(output)
                predictions = tf.cast(predicted_probs > 0.5, tf.float32)
                self._ops[f'prediction_{i}'] = predictions

                #self._ops[f'loss_{i}'] = tf.nn.sigmoid_cross_entropy_with_logits(labels=self._placeholders['output'],
                #                                                                 logits=output)

                self._ops[f'loss_{i}'] = self.classification_loss(predicted_probs=predicted_probs,
                                                                  predictions=predictions,
                                                                  labels=self._placeholders['output'],
                                                                  pos_weight=self.hypers.model_params['pos_weights'][i],
                                                                  neg_weight=self.hypers.model_params['neg_weights'][i])

                self._ops[self.accuracy_op_names[i]] = tf.reduce_mean(1.0 - tf.abs(predictions - self._placeholders['output']))
                output = predictions
            else:
                self._ops[f'prediction_{i}'] = output
                self._ops[f'loss_{i}'] = tf.reduce_sum(tf.square(output - self._placeholders['output']), axis=-1)  # B

            self._ops[f'gates_{i}'] = gates.stack()  # B x T x M x D
            self._ops[f'states_{i}'] = states.stack()  # B x T x D

            outputs.append(output)

        combined_outputs = tf.concat(tf.nest.map_structure(lambda t: tf.expand_dims(t, axis=1), outputs), axis=1)
        self._ops['predictions'] = combined_outputs  # B x N x D'
        self._loss_ops = self._make_loss_ops(use_previous_layers=True)

    def make_loss(self):
        losses: List[tf.Tensor] = []

        # The loss_op keys are ordered by the output level
        for loss_op_name in self._loss_ops:
            losses.append(tf.reduce_mean(self._ops[loss_op_name]))

        losses = tf.stack(losses)  # N, N is the number of sequences
        weighted_losses = tf.reduce_sum(losses * self._placeholders['loss_weights'], axis=-1)  # Scalar

        self._ops['loss'] = weighted_losses

    def anytime_generator(self, feed_dict: Dict[tf.Tensor, List[Any]],
                          max_num_levels: int) -> Optional[Iterable[np.ndarray]]:
        """
        Anytime Inference in a generator-like fashion
        """
        with self.sess.graph.as_default():
            # Initialize partial run settings
            prediction_ops = [self._ops[op_name] for op_name in self.prediction_ops]
            placeholders = list(self._placeholders.values())
            handle = self.sess.partial_run_setup(prediction_ops, placeholders)

            result = None
            for level in range(max_num_levels):
                prediction_op = prediction_ops[level]
                op_name = self.prediction_ops[level]

                # Filter feed dict to avoid feeding the same inputs multiple times
                op_feed_dict = {key: value for key, value in feed_dict.items() if key.name.endswith(str(level))}
                prediction = self.sess.partial_run(handle, prediction_op, feed_dict=op_feed_dict)

                yield prediction

    def anytime_inference(self, feed_dict: Dict[tf.Tensor, List[Any]],
                          max_num_levels: int,
                          processing_params: Dict[str, float],
                          system_energy: float,
                          period_time: float,
                          inference_time: float,
                          max_energy: float,
                          min_energy: float,
                          recharge_rate: float) -> Tuple[Optional[np.ndarray], float, float, int]:
        """
        Executes anytime inference with the given inference and system parameters.
        """
        remaining_time = inference_time - period_time
        with self.sess.graph.as_default():

            # Initialize partial run settings
            prediction_ops = [self._ops[op_name] for op_name in self.prediction_ops]
            placeholders = list(self._placeholders.values())
            handle = self.sess.partial_run_setup(prediction_ops, placeholders)

            prediction_level = 0
            result = None
            while prediction_level < max_num_levels:
                prediction_op = prediction_ops[prediction_level]
                op_name = self.prediction_ops[prediction_level]

                # Filter feed dict to avoid feeding the same inputs multiple times
                op_feed_dict = {key: value for key, value in feed_dict.items() if key.name.endswith(str(prediction_level))}
                prediction = self.sess.partial_run(handle, prediction_op, feed_dict=op_feed_dict)

                step_energy = max(processing_params['energy'] + np.random.normal(loc=0.0, scale=processing_params['energy_noise']), 0.0)
                step_time = max(processing_params['time'] + np.random.normal(loc=0.0, scale=processing_params['time_noise']), 0.0)

                # Accumulate period time
                period_time += step_time

                # There is not enough energy to compute this result
                if system_energy - step_energy < min_energy:
                    if prediction_level == 0:
                        return None, min_energy, period_time, 0
                    return result, min_energy, period_time, prediction_level + 1

                # Safely deduct from the system energy
                system_energy -= step_energy

                # There is not enough time to compute this result
                energy_delta = max_energy - system_energy
                time_delta = inference_time - period_time
                
                if time_delta * recharge_rate <= energy_delta:
                    if prediction_level == 0:
                        result = prediction
                        prediction_level += 1

                    # On an intermittent system, we might want to assume we can interrupt progress at any time
                    # Thus, we set the period time as if we halted execution right when we see
                    # time_delta * recharge_rate == energy_delta. Note this for later.

                    break

                # Advance the prediction level
                result = prediction
                prediction_level += 1

        return result, system_energy, period_time, prediction_level

    def _make_loss_ops(self, use_previous_layers: bool = True) -> OrderedDict:
        loss_ops = OrderedDict()
        var_prefixes: Dict[str, float] = dict()  # Maps prefix name to current weight
        trainable_vars = self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        layer_weight = self.hypers.model_params['layer_weight']

        for i in range(self.num_outputs):
            layer_name = f'layer-{i}'
            var_prefixes[layer_name] = 1.0

            layer_vars = var_filter(trainable_vars, var_prefixes)
            if self.hypers.model_params['share_cell_weights'] or self.model_type in (RNNModelType.CASCADE, RNNModelType.SINGLE):
                # Always include the RNN cell weights
                cell_vars = [VariableWithWeight(var, 1.0) for var in trainable_vars if 'rnn-cell' in var.name]
                layer_vars.extend(cell_vars)

            loss_ops[f'loss_{i}'] = layer_vars

            if not use_previous_layers:
                var_prefixes.pop(layer_name)

            if layer_weight < 1.0:
                # Update the layer weights
                for prefix in var_prefixes:
                    var_prefixes[prefix] = var_prefixes[prefix] * layer_weight

        return loss_ops
