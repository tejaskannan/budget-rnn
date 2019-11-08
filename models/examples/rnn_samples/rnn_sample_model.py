import tensorflow as tf
import numpy as np
import re
import time
from enum import Enum, auto
from collections import namedtuple, defaultdict, OrderedDict
from typing import List, Optional, Tuple, Dict, Any, Set, Union, DefaultDict
from sklearn.preprocessing import StandardScaler
from dpu_utils.utils import RichPath

from models.base_model import Model, VariableWithWeight
from layers.basic import rnn_cell, mlp
from dataset.dataset import Dataset, DataSeries
from utils.hyperparameters import HyperParameters
from utils.tfutils import pool_rnn_outputs


VAR_NAME_REGEX = re.compile(r'.*(layer-[0-9])+.*')
TestMetrics = namedtuple('TestMetrics', ['raw', 'mean', 'std', 'median', 'first_quartile', 'third_quartile'])


class RNNModelType:
    VANILLA = auto()
    SAMPLE = auto()
    CASCADE = auto()
    CHUNKED = auto()


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
    def samples_per_seq(self) -> int:
        seq_length = self.metadata['seq_length']
        return int(seq_length * self.hypers.model_params['sample_frac'])

    def load_metadata(self, dataset: Dataset):
        input_samples: List[List[float]] = []
        output_samples: List[List[float]] = []

        for sample in dataset.dataset[DataSeries.TRAIN]:
            input_samples.append(sample['inputs'])
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
                bin_bounds.append((np.min(split), np.max(split)))
                bin_means.append(np.average(split))

            self.metadata['bin_bounds'] = bin_bounds
            self.metadata['bin_means'] = bin_means

        self.metadata['input_scaler'] = input_scaler
        self.metadata['output_scaler'] = output_scaler
        self.metadata['num_input_features'] = num_input_features
        self.metadata['num_output_features'] = num_output_features
        self.metadata['seq_length'] = seq_length

    def batch_to_feed_dict(self, batch: Dict[str, List[Any]], is_train: bool) -> Dict[tf.Tensor, np.ndarray]:
        dropout = self.hypers.model_params.get('dropout_keep_rate', 1.0) if is_train else 1.0
        input_batch = np.array(batch['inputs'])
        output_batch = np.array(batch['output'])

        input_batch = np.squeeze(input_batch, axis=1)

        num_input_features = self.metadata['num_input_features']
        num_output_features = self.metadata['num_output_features']

        feed_dict = {
            self._placeholders['output']: output_batch.reshape(-1, num_output_features),
            self._placeholders['dropout_keep_rate']: dropout
        }

        if self.hypers.model_params['bin_outputs']:
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
            if self.model_type == RNNModelType.VANILLA:
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
            if self.model_type == RNNModelType.VANILLA:
                samples_per_seq += self.samples_per_seq

        self._placeholders['output'] = tf.placeholder(shape=[None, num_output_features],
                                                      dtype=tf.float32,
                                                      name='output')  # B x K
        self._placeholders['dropout_keep_rate'] = tf.placeholder(shape=[], dtype=tf.float32, name='dropout-keep-rate')
        self._placeholders['loss_weights'] = tf.placeholder(shape=[self.num_outputs],
                                                            dtype=tf.float32,
                                                            name='loss-weights')

        if self.hypers.model_params['bin_outputs']:
            self._placeholders['bin_means'] = tf.placeholder(shape=[None, self.hypers.model_params['num_bins']],
                                                             dtype=tf.float32,
                                                             name='bin-means')


    def predict(self, dataset: Dataset, name: str,
                test_batch_size: Optional[int] = None,
                max_num_batches: Optional[int] = None) -> Tuple[Dict[str, TestMetrics], Dict[str, TestMetrics], List[np.array]]:
        test_batch_size = test_batch_size if test_batch_size is not None else self.hypers.batch_size
        test_batch_generator = dataset.minibatch_generator(series=DataSeries.TEST,
                                                           batch_size=test_batch_size,
                                                           metadata=self.metadata,
                                                           should_shuffle=False,
                                                           drop_incomplete_batches=True)

        prediction_ops = self.prediction_ops

        mse_dict: DefaultDict[str, List[float]] = defaultdict(list)
        latency_dict: DefaultDict[str, List[float]] = defaultdict(list)

        true_values: List[np.array] = []

        num_batches = 0
        for batch in test_batch_generator:
            feed_dict = self.batch_to_feed_dict(batch, is_train=False)
            op_results: Dict[str, Any] = dict()

            # Execute operations individually for better profiling
            for i in range(len(prediction_ops)):
                ops_to_run = prediction_ops[0:i+1]

                start = time.time()
                result = self.execute(feed_dict, ops_to_run)
                elapsed = time.time() - start

                # Save results
                op = prediction_ops[i]
                op_results[op] = result[op]

                # Do not accumulate latency metrics on first batch (to avoid outliers from caching)
                if num_batches > 0:
                    latency_dict[op].append(elapsed * 1000.0)

            for prediction_op in prediction_ops:
                prediction = op_results[prediction_op]

                unnormalized_prediction = self.metadata['output_scaler'].inverse_transform(prediction)

                raw_expected = np.array(batch['output']).reshape(-1, self.metadata['num_output_features'])
                expected = self.metadata['output_scaler'].inverse_transform(raw_expected)
                true_values.append(expected)

                squared_error = np.sum(np.square(expected - unnormalized_prediction), axis=-1)

                # Avoid the first batch for consistency with latency measurements
                if num_batches > 0:
                    mse_dict[prediction_op].extend(squared_error)

            num_batches += 1

            if max_num_batches is not None and num_batches >= max_num_batches:
                break

        # Compute aggregate statistics
        mse_metrics: Dict[str, TestMetrics] = dict()
        latency_metrics: Dict[str, TestMetrics] = dict()
        for prediction_op in prediction_ops:
            raw_errors = mse_dict[prediction_op]
            error = TestMetrics(raw=raw_errors,
                                mean=np.average(raw_errors),
                                std=np.std(raw_errors),
                                median=np.median(raw_errors),
                                first_quartile=np.percentile(raw_errors, 25),
                                third_quartile=np.percentile(raw_errors, 75))

            raw_latency = latency_dict[prediction_op]
            latency = TestMetrics(raw=np.repeat(raw_latency, test_batch_size, axis=0),
                                  mean=np.average(raw_latency),
                                  std=np.std(raw_latency),
                                  median=np.median(raw_latency),
                                  first_quartile=np.percentile(raw_latency, 25),
                                  third_quartile=np.percentile(raw_latency, 75))

            mse_metrics[prediction_op] = error
            latency_metrics[prediction_op] = latency

        return mse_metrics, latency_metrics, true_values

    def make_model(self, is_train: bool):
        with tf.variable_scope('rnn-model', reuse=tf.AUTO_REUSE):
            if self.model_type == RNNModelType.SAMPLE:
                self.make_rnn_sample_model(is_train)
            else:
                self.make_rnn_model(is_train)

    def make_rnn_model(self, is_train: bool):
        rnn_layers = self.hypers.model_params['rnn_layers']
        outputs: List[tf.Tensor] = []
        prev_state: Optional[tf.Tensor] = None

        for i in range(self.num_sequences):
            rnn_cell_name = 'rnn-cell'
            output_layer_name = 'output-layer'
            rnn_model_name = 'rnn-model'

            if self.model_type == RNNModelType.CASCADE:
                # The Cascade model shares cell weights by design
                output_layer_name = f'{output_layer_name}-layer-{i}'
            elif not self.hypers.model_params.get('share_cell_weights', False):
                rnn_cell_name = f'{rnn_cell_name}-layer-{i}'
                output_layer_name = f'{output_layer_name}-layer-{i}'
                rnn_model_name = f'{rnn_model_name}-layer-{i}'

            cell = rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                            num_units=self.hypers.model_params['state_size'],
                            activation=self.hypers.model_params['rnn_activation'],
                            dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                            num_layers=rnn_layers,
                            state_is_tuple=False,
                            name=rnn_cell_name,
                            dtype=tf.float32)

            inputs = self._placeholders[f'input_{i}']
            initial_state = cell.zero_state(batch_size=tf.shape(inputs)[0], dtype=tf.float32)

            if self.model_type in (RNNModelType.CHUNKED, RNNModelType.CASCADE) and prev_state is not None:
                initial_state = prev_state

            rnn_outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                                   inputs=inputs,
                                                   initial_state=initial_state,
                                                   dtype=tf.float32,
                                                   scope=rnn_model_name)

            # Save previous state for the chunked model
            if self.model_type in (RNNModelType.CHUNKED, RNNModelType.CASCADE):
                prev_state = state

            # B x D
            rnn_output = pool_rnn_outputs(rnn_outputs, state, pool_mode=self.hypers.model_params['pool_mode'])

            # B x D'
            if self.hypers.model_params['bin_outputs']:
                num_output_features = len(self.metadata['bin_means'])
            else:
                num_output_features = self.metadata['num_output_features']

            output = mlp(inputs=rnn_output,
                         output_size=num_output_features,
                         hidden_sizes=self.hypers.model_params.get('output_hidden_units'),
                         activations=self.hypers.model_params['output_hidden_activation'],
                         dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                         name=output_layer_name)

            if self.hypers.model_params['bin_outputs']:
                output_probs = tf.nn.softmax(output, axis=-1)
                output = tf.reduce_sum(output_probs * self._placeholders['bin_means'], axis=-1, keepdims=True)
                self._ops[f'prediction_probs_{i}'] = output_probs

            self._ops[f'prediction_{i}'] = output
            outputs.append(output)

        self._ops['predictions'] = tf.concat([tf.expand_dims(t, axis=1) for t in outputs], axis=1)
        self._loss_ops = self._make_loss_ops(use_previous_layers=False)

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

            cell = rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                            num_units=self.hypers.model_params['state_size'],
                            activation=self.hypers.model_params['rnn_activation'],
                            dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                            num_layers=rnn_layers,
                            state_is_tuple=False,
                            name=cell_name,
                            dtype=tf.float32)

            prev_states = states_list[i-1] if i > 0 else None
            rnn_outputs, states = self.__dynamic_rnn(inputs=self._placeholders[f'input_{i}'],
                                                     cell=cell,
                                                     previous_states=prev_states,
                                                     name=f'rnn-layer-{i}')

            last_index = tf.shape(self._placeholders[f'input_{i}'])[1] - 1
            final_state = states.read(last_index)
            states_list.append(states)

            rnn_output = pool_rnn_outputs(rnn_outputs, final_state, pool_mode=self.hypers.model_params['pool_mode'])

            if self.hypers.model_params['bin_outputs']:
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

            if self.hypers.model_params['bin_outputs']:
                output_probs = tf.nn.softmax(output, axis=-1)
                output = tf.reduce_sum(output_probs * self._placeholders['bin_means'], axis=-1, keepdims=True)
                self._ops[f'prediction_probs_{i}'] = output_probs

            self._ops[f'prediction_{i}'] = output

            self._ops[f'loss_{i}'] = tf.reduce_sum(tf.square(output - self._placeholders['output']), axis=-1)  # B
            outputs.append(output)

        combined_outputs = tf.concat([tf.expand_dims(t, axis=1) for t in outputs], axis=1)
        self._ops['predictions'] = combined_outputs  # B x N x D'
        self._loss_ops = self._make_loss_ops()

    def make_loss(self):
        losses: List[tf.Tensor] = []

        # The loss_op keys are ordered by the output level
        for loss_op_name in self._loss_ops:
            losses.append(self._ops[loss_op_name])

        losses = tf.stack(losses)  # B x N, N is the number of sequences
        loss_weights = tf.expand_dims(self._placeholders['loss_weights'], axis=1)
        weighted_losses = tf.reduce_sum(losses * loss_weights, axis=-1)  # B

        self._ops['loss'] = tf.reduce_mean(weighted_losses)  # Weighted average over all layers

    def _make_loss_ops(self, use_previous_layers: bool = True) -> OrderedDict:
        loss_ops = OrderedDict()
        var_prefixes: Dict[str, float] = dict()  # Maps prefix name to current weight
        trainable_vars = self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        layer_weight = self.hypers.model_params['layer_weight']

        for i in range(self.num_outputs):
            layer_name = f'layer-{i}'
            var_prefixes[layer_name] = 1.0

            layer_vars = var_filter(trainable_vars, var_prefixes)
            if self.hypers.model_params['share_cell_weights'] or self.model_type == RNNModelType.CASCADE:
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

    def __dynamic_rnn(self,
                      inputs: tf.Tensor,
                      cell: tf.nn.rnn_cell.BasicRNNCell,
                      previous_states: Optional[tf.TensorArray] = None,
                      name: str = None) -> Tuple[tf.TensorArray, tf.TensorArray]:
        state_size = self.hypers.model_params['state_size']
        rnn_layers = self.hypers.model_params['rnn_layers']

        states_array = tf.TensorArray(dtype=tf.float32, size=self.samples_per_seq, dynamic_size=False, clear_after_read=False)
        outputs_array = tf.TensorArray(dtype=tf.float32, size=self.samples_per_seq, dynamic_size=False)

        if previous_states is not None:
            combine_layer_name = 'combine-states' if name is None else f'{name}-combine-states'
            units = (rnn_layers * state_size) if self.hypers.model_params.get('combine_element_wise', False) else 1
            combine_states = tf.layers.Dense(units=units,
                                             activation=tf.math.sigmoid,
                                             use_bias=True,
                                             kernel_initializer=tf.initializers.glorot_uniform(),
                                             name=combine_layer_name)

        # While loop step
        def step(index, state, outputs, states):
            step_inputs = tf.gather(inputs, indices=index, axis=1)  # B x D

            combined_state = state
            if previous_states is not None:
                prev_state = previous_states.read(index)

                combine_weight = combine_states(tf.concat([state, prev_state], axis=-1))
                combined_state = combine_weight * state + (1.0 - combine_weight) * prev_state

            output, state = cell(step_inputs, combined_state)
            outputs = outputs.write(index=index, value=output)
            states = states.write(index=index, value=state)

            return [index + 1, state, outputs, states]

        def cond(index, _1, _2, _3):
            return index < self.samples_per_seq

        i = tf.constant(0, dtype=tf.int32)
        state = cell.zero_state(batch_size=tf.shape(inputs)[0], dtype=tf.float32)
        _, _, final_outputs, final_states = tf.while_loop(cond=cond,
                                                          body=step,
                                                          loop_vars=[i, state, outputs_array, states_array],
                                                          parallel_iterations=1,
                                                          maximum_iterations=self.samples_per_seq,
                                                          name='rnn-while-loop')
        return final_outputs, final_states
