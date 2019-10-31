import tensorflow as tf
import numpy as np
import re
import time
from enum import Enum, auto
from collections import OrderedDict
from typing import List, Optional, Tuple, Dict, Any, Set, Union
from sklearn.preprocessing import StandardScaler
from dpu_utils.utils import RichPath

from models.base_model import Model, VariableWithWeight
from layers.basic import rnn_cell
from dataset.dataset import Dataset, DataSeries
from utils.hyperparameters import HyperParameters
from utils.tfutils import pool_rnn_outputs


VAR_NAME_REGEX = re.compile(r'.*(layer-[0-9])+.*')


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

    @property
    def num_sequences(self) -> int:
        if self.model_type == RNNModelType.CASCADE:
            return 1
        return int(1.0 / self.hypers.model_params['sample_frac'])

    @property
    def num_outputs(self) -> int:
        return int(1.0 / self.hypers.model_params['sample_frac'])

    @property
    def samples_per_seq(self) -> int:
        seq_length = self.metadata['seq_length']
        if self.model_type == RNNModelType.CASCADE:
            return seq_length
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
            elif self.model_type == RNNModelType.CHUNKED:
                start, end = i * samples_per_seq, (i+1) * samples_per_seq
                sample_tensor = input_batch[:, start:end]
                feed_dict[self._placeholders[f'input_{i}']] = sample_tensor
            else:  # Cascade model
                feed_dict[self._placeholders[f'input_{i}']] = input_batch

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

    def predict(self, dataset: Dataset, name: str,
                test_batch_size: Optional[int] = None,
                max_num_batches: Optional[int] = None) -> Tuple[OrderedDict, OrderedDict]:
        test_batch_size = test_batch_size if test_batch_size is not None else self.hypers.batch_size
        test_batch_generator = dataset.minibatch_generator(series=DataSeries.TEST,
                                                           batch_size=test_batch_size,
                                                           metadata=self.metadata,
                                                           should_shuffle=False,
                                                           drop_incomplete_batches=True)

        prediction_ops = [f'prediction_{i}' for i in range(self.num_outputs)]
        mse_dict = {prediction_op: 0.0 for prediction_op in prediction_ops}
        latency_dict = {prediction_op: 0.0 for prediction_op in prediction_ops}
        
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
                    latency_dict[op] += elapsed

            for prediction_op in prediction_ops:
                prediction = op_results[prediction_op]
 
                unnormalized_prediction = self.metadata['output_scaler'].inverse_transform(prediction)

                raw_expected = np.array(batch['output']).reshape(-1, self.metadata['num_output_features'])
                expected = self.metadata['output_scaler'].inverse_transform(raw_expected)

                squared_error = np.sum(np.square(expected - unnormalized_prediction), axis=-1)
                mse = np.average(squared_error)
                mse_dict[prediction_op] += mse

            num_batches += 1

            if num_batches >= max_num_batches:
                break

        for op in mse_dict.keys():
            latency_dict[op] = latency_dict[op] / (float(num_batches - 1) + 1e-7)
            mse_dict[op] = mse_dict[op] / (float(num_batches) + 1e-7)

        ordered_mse_dict = OrderedDict(sorted(mse_dict.items(), key=lambda t: t[0]))
        ordered_latency_dict = OrderedDict(sorted(latency_dict.items(), key=lambda t: t[0]))

        return ordered_mse_dict, ordered_latency_dict

    def make_model(self, is_train: bool):
        with tf.variable_scope('rnn-model', reuse=tf.AUTO_REUSE):
            
            if self.model_type in (RNNModelType.VANILLA, RNNModelType.CHUNKED):
                self.make_rnn_model(is_train)
            elif self.model_type == RNNModelType.SAMPLE:
                self.make_rnn_sample_model(is_train)
            else:  # Cascade model
                self.make_rnn_cascade_model(is_train)
            
    def make_rnn_model(self, is_train: bool):
        rnn_cell_name = 'rnn-cell'
        output_layer_name = 'output-layer'
        rnn_model_name = 'rnn-model'
        outputs: List[tf.Tensor] = []
        prev_state: Optional[tf.Tensor] = None

        for i in range(self.num_sequences):

            if not self.hypers.model_params.get('share_cell_weights', False):
                rnn_cell_name = f'{rnn_cell_name}-layer-{i}'
                output_layer_name = f'{output_layer_name}-layer-{i}'
                rnn_model_name = f'{rnn_model_name}-layer-{i}'

            cell = rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                            num_units=self.hypers.model_params['state_size'],
                            activation=self.hypers.model_params.get('rnn_activation', 'tanh'),
                            dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                            name=rnn_cell_name,
                            dtype=tf.float32)

            inputs = self._placeholders[f'input_{i}']
            initial_state = cell.zero_state(batch_size=tf.shape(inputs)[0], dtype=tf.float32)

            if self.model_type == RNNModelType.CHUNKED and prev_state is not None:
                initial_state = prev_state

            rnn_outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                                   inputs=inputs,
                                                   initial_state=initial_state,
                                                   dtype=tf.float32,
                                                   scope=rnn_model_name)

            # B x D
            rnn_output = pool_rnn_outputs(rnn_outputs, state, pool_mode=self.hypers.model_params['pool_mode'])

            # Set previous state if using the chunked model
            if self.model_type == RNNModelType.CHUNKED:
                prev_state = state

            # B x D'
            output = tf.layers.dense(inputs=rnn_output,
                                     units=self.metadata['num_output_features'],
                                     activation=None,
                                     kernel_initializer=tf.initializers.glorot_uniform(),
                                     name=output_layer_name)
            self._ops[f'prediction_{i}'] = output
            self._ops[f'loss_{i}'] = tf.reduce_sum(tf.square(output - self._placeholders['output']), axis=-1)  # B
            outputs.append(output)

        self._ops['predictions'] = tf.concat([tf.expand_dims(t, axis=1) for t in outputs], axis=1)
        self._loss_ops = self._make_loss_ops()

    def make_rnn_sample_model(self, is_train: bool):
        outputs: List[tf.Tensor] = []
        states_list: List[tf.TensorArray] = []
        num_sequences = int(1.0 / self.hypers.model_params['sample_frac'])

        for i in range(num_sequences):
            # Create the RNN Cell
            cell_name = 'rnn-cell'
            if not self.hypers.model_params['share_cell_weights']:
                cell_name = f'{cell_name}-layer-{i}'  # If not weight sharing, then each cell has its own scope

            cell = rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                            num_units=self.hypers.model_params['state_size'],
                            activation=self.hypers.model_params.get('rnn_activation', 'tanh'),
                            dropout_keep_rate=self._placeholders['dropout_keep_rate'],
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

            # B x D'
            output = tf.layers.dense(inputs=rnn_output,
                                     units=self.metadata['num_output_features'],
                                     activation=None,
                                     use_bias=False,
                                     kernel_initializer=tf.initializers.glorot_uniform(),
                                     name=f'output-layer-{i}')
            self._ops[f'prediction_{i}'] = output
            self._ops[f'loss_{i}'] = tf.reduce_sum(tf.square(output - self._placeholders['output']), axis=-1)  # B

            outputs.append(output)

        combined_outputs = tf.concat([tf.expand_dims(t, axis=1) for t in outputs], axis=1)
        self._ops['predictions'] = combined_outputs  # B x N x D'
        self._loss_ops = self._make_loss_ops()

    def make_rnn_cascade_model(self, is_train: bool):
        # Make the RNN cell
        cell = rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                        num_units=self.hypers.model_params['state_size'],
                        activation=self.hypers.model_params.get('rnn_activation', 'tanh'),
                        dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                        name='rnn-cell',
                        dtype=tf.float32)
 
        # Run the RNN
        rnn_outputs, states = self.__dynamic_rnn(inputs=self._placeholders[f'input_0'],
                                                 cell=cell,
                                                 name='rnn')

        # Transform outputs
        step = int(self.metadata['seq_length'] / self.num_outputs)
        outputs: List[tf.Tensor] = []
        for i, seq_index in enumerate(range(step - 1, self.metadata['seq_length'], step)):
            state = states.read(index=seq_index)

            output = tf.layers.dense(inputs=state,
                                     units=self.metadata['num_output_features'],
                                     activation=None,
                                     use_bias=False,
                                     kernel_initializer=tf.initializers.glorot_uniform(),
                                     name=f'output-layer-{i}')
            self._ops[f'prediction_{i}'] = output
            self._ops[f'loss_{i}'] = tf.reduce_sum(tf.square(output - self._placeholders['output']), axis=-1)  # B
            outputs.append(output)

        self._ops['predictions'] = tf.concat([tf.expand_dims(t, axis=1) for t in outputs], axis=1)
        self._loss_ops = self._make_loss_ops(use_previous_layers=False)

    def make_loss(self):
        losses: List[tf.Tensor] = []
        for loss_op_name in self._loss_ops:
            losses.append(self._ops[loss_op_name])

        losses = tf.stack(losses)  # B x N, N is the number of sequences
        loss_weights = tf.expand_dims(self._placeholders['loss_weights'], axis=1)
        weighted_losses = tf.reduce_sum(losses * loss_weights, axis=-1)  # B

        self._ops['loss'] = tf.reduce_mean(weighted_losses)  # Weighted average over all layers
 
    def _make_loss_ops(self, use_previous_layers: bool = True) -> Dict[str, List[VariableWithWeight]]:
        loss_ops: Dict[str, List[VariableWithWeight]] = dict()
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

        states_array = tf.TensorArray(dtype=tf.float32, size=self.samples_per_seq, dynamic_size=False, clear_after_read=False)
        outputs_array = tf.TensorArray(dtype=tf.float32, size=self.samples_per_seq, dynamic_size=False)

        if previous_states is not None:
            combine_layer_name = 'combine-states' if name is None else f'{name}-combine-states'
            combine_states = tf.layers.Dense(units=1,
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
