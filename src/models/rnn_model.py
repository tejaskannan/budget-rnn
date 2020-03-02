import tensorflow as tf

import numpy as np
import re
import time
from collections import namedtuple, defaultdict, OrderedDict
from typing import List, Optional, Tuple, Dict, Any, Set, Union, DefaultDict, Iterable
from sklearn.preprocessing import StandardScaler
from dpu_utils.utils import RichPath

from models.base_model import Model, VariableWithWeight
from layers.basic import rnn_cell, mlp
from layers.cells.cells import make_rnn_cell, MultiRNNCell
from layers.rnn import dynamic_rnn, dropped_rnn
from layers.output_layers import OutputType, compute_binary_classification_output, compute_regression_output
from dataset.dataset import Dataset, DataSeries
from utils.hyperparameters import HyperParameters
from utils.tfutils import pool_rnn_outputs
from utils.constants import SMALL_NUMBER
from utils.rnn_utils import *
from rnn_samples.testing_utils import TestMetrics, Prediction


VAR_NAME_REGEX = re.compile(r'.*(layer-[0-9])+.*')


def var_filter(trainable_vars: List[tf.Variable], var_prefixes: Optional[Dict[str, float]] = None) -> List[VariableWithWeight]:
    result: List[VariableWithWeight] = []
    for var in trainable_vars:
        name_match = VAR_NAME_REGEX.match(var.name)
        layer_name = name_match.group(1) if name_match is not None else None
        if var_prefixes is None or layer_name in var_prefixes:
            weight = 1.0 if var_prefixes is None else var_prefixes[layer_name]
            result.append(VariableWithWeight(var, weight))
    return result


class RNNModel(Model):

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
        self.__output_type = OutputType[self.hypers.model_params['output_type'].upper()]

    @property
    def sample_frac(self) -> float:
        return self.hypers.model_params['sample_frac']

    @property
    def num_sequences(self) -> int:
        return int(1.0 / self.sample_frac)

    @property
    def num_outputs(self) -> int:
        return int(1.0 / self.sample_frac)

    @property
    def prediction_ops(self) -> List[str]:
        return [get_prediction_name(i) for i in range(self.num_outputs)]

    @property
    def accuracy_op_names(self) -> List[str]:
        return [get_accuracy_name(i) for i in range(self.num_outputs)]

    @property
    def samples_per_seq(self) -> int:
        seq_length = self.metadata['seq_length']
        return int(seq_length * self.sample_frac)

    @property
    def output_type(self) -> OutputType:
        return self.__output_type

    @property
    def output_name(self) -> str:
        return 'output'

    @property
    def loss_op_names(self) -> List[str]:
        if self.model_type == RNNModelType.VANILLA:
            return [get_loss_name(i) for i in range(self.num_outputs)]
        return ['loss']

    def get_variable_group(self, loss_op_name: str) -> List[tf.Variable]:
        variables: List[tf.Variable] = []
        trainable_vars = list(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

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

        # Fetch training samples to prepare for normalization
        for sample in dataset.dataset[DataSeries.TRAIN]:
            input_sample = np.array(sample['inputs'])
            input_samples.append(input_sample)

            if not isinstance(sample['output'], list) and \
                    not isinstance(sample['output'], np.ndarray):
                output_samples.append([sample['output']])
            else:
                output_samples.append(sample['output'])

        # Infer the number of input and output features
        num_input_features = len(input_samples[0][0])
        seq_length = len(input_samples[0])
        input_samples = np.reshape(input_samples, newshape=(-1, num_input_features))

        input_scaler = StandardScaler()
        input_scaler.fit(input_samples)

        num_output_features = len(output_samples[0])
        output_scaler = StandardScaler()
        output_scaler.fit(output_samples)

        self.metadata['input_scaler'] = input_scaler
        self.metadata['output_scaler'] = output_scaler
        self.metadata['num_input_features'] = num_input_features
        self.metadata['num_output_features'] = num_output_features
        self.metadata['seq_length'] = seq_length
        self.metadata['should_normalize_output'] = self.output_type == OutputType.REGRESSION

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
            input_ph = self.placeholders[get_input_name(i)]

            if self.model_type in (RNNModelType.VANILLA, RNNModelType.DROPPED, RNNModelType.SINGLE):
                seq_indexes.extend(range(i, seq_length, num_sequences))
                seq_indexes = list(sorted(seq_indexes))
                sample_tensor = input_batch[:, seq_indexes]
                feed_dict[input_ph] = sample_tensor
            elif self.model_type == RNNModelType.SAMPLE:
                seq_indexes = list(range(i, seq_length, num_sequences))
                sample_tensor = input_batch[:, seq_indexes]
                feed_dict[input_ph] = sample_tensor
            else:  # Chunked or Cascade
                start, end = i * samples_per_seq, (i+1) * samples_per_seq
                sample_tensor = input_batch[:, start:end]
                feed_dict[input_ph] = sample_tensor

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

            # [B, S, D]
            self._placeholders[get_input_name(i)] = tf.placeholder(shape=input_shape,
                                                                   dtype=tf.float32,
                                                                   name=get_input_name(i))
            if self.model_type in (RNNModelType.VANILLA, RNNModelType.DROPPED, RNNModelType.SINGLE):
                samples_per_seq += self.samples_per_seq

        # [B, K]
        self._placeholders[self.output_name] = tf.placeholder(shape=[None, num_output_features],
                                                              dtype=tf.float32,
                                                              name=self.output_name)
        self._placeholders['dropout_keep_rate'] = tf.placeholder(shape=[],
                                                                 dtype=tf.float32,
                                                                 name='dropout-keep-rate')
        self._placeholders['loss_weights'] = tf.placeholder(shape=[self.num_outputs],
                                                            dtype=tf.float32,
                                                            name='loss-weights')

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
            self.make_rnn_model(is_train)

    def make_rnn_model(self, is_train: bool):
        outputs: List[tf.Tensor] = []
        states_list: List[tf.TensorArray] = []
        prev_state: Optional[tf.Tensor] = None

        num_output_features = self.metadata['num_output_features']

        for i in range(self.num_sequences):
            # Get relevant variable names
            input_name = get_input_name(i)
            cell_name = get_cell_level_name(i, self.hypers.model_params['share_cell_weights'])
            rnn_level_name = get_rnn_level_name(i)
            output_layer_name = get_output_layer_name(i)
            logits_name = get_logits_name(i)
            prediction_name = get_prediction_name(i)
            loss_name = get_loss_name(i)
            gate_name = get_gates_name(i)
            state_name = get_states_name(i)
            accuracy_name = get_accuracy_name(i)

            # Create the RNN Cell
            cell = make_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                 input_units=self.metadata['num_input_features'],
                                 output_units=self.hypers.model_params['state_size'],
                                 activation=self.hypers.model_params['rnn_activation'],
                                 dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                                 num_layers=self.hypers.model_params['rnn_layers'],
                                 name=cell_name)

            inputs = self._placeholders[input_name]
            initial_state = cell.zero_state(batch_size=tf.shape(inputs)[0], dtype=tf.float32)

            # Set the initial state for chunked model types
            if self.model_type == RNNModelType.CASCADE and prev_state is not None:
                initial_state = prev_state

            # Set previous states for the Sample model type
            prev_states = None
            if self.model_type == RNNModelType.SAMPLE and i > 0:
                prev_states = states_list[i-1]

            rnn_outputs, rnn_states, rnn_gates = dynamic_rnn(cell=cell,
                                                             inputs=inputs,
                                                             previous_states=prev_states,
                                                             initial_state=initial_state,
                                                             name=rnn_level_name)
            # Save previous states
            states_list.append(rnn_states)

            # Get the final state
            last_index = tf.shape(inputs)[1] - 1
            final_output = rnn_outputs.read(index=last_index)

            # Save previous state for the chunked model
            if self.model_type == RNNModelType.CASCADE:
                prev_state = rnn_states.read(index=last_index)

            # [B, D]
            rnn_output = pool_rnn_outputs(rnn_outputs, final_output, pool_mode=self.hypers.model_params['pool_mode'])

            # [B, K]
            output = mlp(inputs=rnn_output,
                         output_size=num_output_features,
                         hidden_sizes=self.hypers.model_params.get('output_hidden_units'),
                         activations=self.hypers.model_params['output_hidden_activation'],
                         dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                         name=output_layer_name)

            if self.output_type == OutputType.CLASSIFICATION:
                classification_output = compute_binary_classification_output(model_output=output,
                                                                             labels=self._placeholders['output'],
                                                                             false_pos_weight=self.hypers.model_params['pos_weights'][i],
                                                                             false_neg_weight=self.hypers.model_params['neg_weights'][i])
                self._ops[logits_name] = classification_output.logits,
                self._ops[prediction_name] = classification_output.predictions
                self._ops[loss_name] = classification_output.loss
                self._ops[accuracy_name] = classification_output.accuracy
            else:
                regression_output = compute_regression_output(model_output=output, expected_otuput=self._placeholders['output'])
                self._ops[prediction_name] = regression_output.predictions
                self._ops[loss_name] = regression_output.loss

            self._ops[gate_name] = rnn_gates.stack()  # [B, T, M, D]
            self._ops[state_name] = rnn_states.stack()  # [B, T, D]

            outputs.append(output)

        combined_outputs = tf.concat(tf.nest.map_structure(lambda t: tf.expand_dims(t, axis=1), outputs), axis=1)
        self._ops[ALL_PREDICTIONS_NAME] = combined_outputs
        use_previous_layers = self.model_type == RNNModelType.CHUNKED
        self._loss_ops = self._make_loss_ops(use_previous_layers=use_previous_layers)

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
        Anytime Inference in a generator-like fashion using Tensorflow's partial run API
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
