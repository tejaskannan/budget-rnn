import tensorflow as tf
import numpy as np
import re
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Dict, Any, Set
from sklearn.preprocessing import StandardScaler

from models.base_model import Model, VariableWithWeight
from layers.basic import rnn_cell
from dataset.dataset import Dataset, DataSeries
from utils.tfutils import pool_rnn_outputs


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


class RNNSampleModel(Model):

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

        feed_dict = {
            self._placeholders['output']: output_batch.reshape(-1, self.metadata['num_output_features']),
            self._placeholders['dropout_keep_rate']: dropout
        }

        # Create subsequences
        seq_length = self.metadata['seq_length']
        sample_frac = self.hypers.model_params['sample_frac']

        samples_per_seq = int(seq_length * sample_frac)
        step = int(1.0 / sample_frac)

        if self.hypers.model_params.get('model_type') == 'vanilla_rnn_model':
            num_sequences = 1
        else:
            num_sequences = step

        loss_weights = self.hypers.model_params.get('loss_weights')
        if loss_weights is None:
            loss_weights = np.ones(shape=num_sequences, dtype=np.float32)

        assert len(loss_weights) == num_sequences, f'Loss weights ({len(loss_weights)}) must match the number of sequences ({num_sequences}).'
        feed_dict[self._placeholders['loss_weights']] = loss_weights / (np.sum(loss_weights) + 1e-7)  # Normalize the loss weights

        for i in range(num_sequences):
            sample_tensor = np.array([input_batch[:, j] for j in range(i, input_batch.shape[1], step)])
            sample_tensor = sample_tensor[:samples_per_seq, :, :]  # Even number of samples per sequence
            sample_tensor = np.transpose(sample_tensor, axes=[1, 0, 2])
            feed_dict[self._placeholders[f'input_{i}']] = sample_tensor
 
        return feed_dict

    def make_placeholders(self):
        num_features = self.metadata['num_input_features']
        seq_length = self.metadata['seq_length']
        sample_frac = self.hypers.model_params['sample_frac']

        # Create placeholders for each sequence
        samples_per_seq = int(seq_length * sample_frac)
        num_sequences = int(1.0 / sample_frac)
        
        if self.hypers.model_params.get('model_type') == 'vanilla_rnn_model':
            num_sequences = 1

        for i in range(num_sequences):
            # B x S x D
            self._placeholders[f'input_{i}'] = tf.placeholder(shape=[None, samples_per_seq, num_features],
                                                              dtype=tf.float32,
                                                              name=f'input-{i}')

        num_output_features = self.metadata['num_output_features']
        self._placeholders['output'] = tf.placeholder(shape=[None, num_output_features],
                                                      dtype=tf.float32,
                                                      name='output')  # B x K
        self._placeholders['dropout_keep_rate'] = tf.placeholder(shape=[], dtype=tf.float32, name='dropout-keep-rate')

        self._placeholders['loss_weights'] = tf.placeholder(shape=[num_sequences],
                                                            dtype=tf.float32,
                                                            name='loss-weights')

    def predict(self, dataset: Dataset, name: str) -> Dict[Any, Any]:

        test_batch_generator = dataset.minibatch_generator(series=DataSeries.TEST,
                                                           batch_size=self.hypers.batch_size,
                                                           metadata=self.metadata,
                                                           should_shuffle=False,
                                                           drop_incomplete_batches=True)
        
        num_sequences = int(1.0 / self.hypers.model_params['sample_frac'])
        if self.hypers.model_params.get('model_type') == 'vanilla_rnn_model':
            num_sequences = 1
            prediction_ops = ['predictions']
        else:
            prediction_ops = [f'prediction_{i}' for i in range(num_sequences)]

        mse_dict = {prediction_op: 0.0 for prediction_op in prediction_ops}
        num_batches = 0
        for batch in test_batch_generator:
            feed_dict = self.batch_to_feed_dict(batch, is_train=False)
            op_result = self.execute(feed_dict, prediction_ops)

            for prediction_op in prediction_ops:
                prediction = op_result[prediction_op]
 
                unnormalized_prediction = self.metadata['output_scaler'].inverse_transform(prediction)

                raw_expected = np.array(batch['output']).reshape(-1, self.metadata['num_output_features'])
                expected = self.metadata['output_scaler'].inverse_transform(raw_expected)

                squared_error = np.sum(np.square(expected - unnormalized_prediction), axis=-1)
                mse = np.average(squared_error)
                mse_dict[prediction_op] += mse

            num_batches += 1

        # Compute average over all batches
        for prediction_op, error in mse_dict.items():
            mse_dict[prediction_op] = error / float(num_batches)

        # Plot mean errors per level
        with plt.style.context('ggplot'):
            x = list(range(num_sequences))
            y = [float(mse_dict[prediction_op]) for prediction_op in prediction_ops]

            plt.plot(x, y, marker='o', markersize=3)
            
            plt.title('Mean Squared Error on Test Set')
            plt.xlabel('Sample Layer')
            plt.ylabel('Mean Squared Error')

            plt.xticks(ticks=x, labels=x)

            plt.savefig(f'{name}.pdf')
            plt.show()

        return mse_dict

    def make_model(self, is_train: bool):
        with tf.variable_scope('rnn-model', reuse=tf.AUTO_REUSE):
            if self.hypers.model_params.get('model_type') == 'vanilla_rnn_model':
                self.make_rnn_model(is_train)
            else:
                self.make_rnn_sample_model(is_train)

    def make_rnn_model(self, is_train: bool):
        cell = rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                        num_units=self.hypers.model_params['state_size'],
                        activation='tanh',
                        dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                        name='rnn-cell',
                        dtype=tf.float32)
        
        inputs = self._placeholders['input_0']
        initial_state = cell.zero_state(batch_size=tf.shape(inputs)[0], dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                           inputs=inputs,
                                           initial_state=initial_state,
                                           dtype=tf.float32,
                                           scope='rnn-model')
        rnn_output = pool_rnn_outputs(outputs, state, pool_mode=self.hypers.model_params['pool_mode'])

        output = tf.layers.dense(inputs=rnn_output,
                                 units=self.metadata['num_output_features'],
                                 activation=None,
                                 kernel_initializer=tf.initializers.glorot_uniform(),
                                 name='output_layer')
        self._ops['predictions'] = output
        loss = tf.reduce_sum(tf.square(output - self._placeholders['output']))
        
        trainable_vars = self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self._loss_ops = {'loss': var_filter(trainable_vars)}
        self._ops['loss'] = loss

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
                            activation='tanh',
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
                                     name=f'output-layer-layer-{i}')
            self._ops[f'prediction_{i}'] = output
            self._ops[f'loss_{i}'] = tf.reduce_sum(tf.square(output - self._placeholders['output']))

            outputs.append(output)

        combined_outputs = tf.concat([tf.expand_dims(t, axis=1) for t in outputs], axis=1)
        self._ops['predictions'] = combined_outputs  # B x N x D'

        # Make the loss operations dictionary to control layer-wise learning rates
        # For each loss operation.
        loss_ops: Dict[str, List[VariableWithWeight]] = dict()
        var_prefixes: Dict[str, float] = dict()  # Maps prefix name to current weight
        trainable_vars = self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        layer_weight = self.hypers.model_params['layer_weight']
        for i in range(num_sequences):
            layer_name = f'layer-{i}'
            var_prefixes[layer_name] = 1.0

            layer_vars = var_filter(trainable_vars, var_prefixes)
            if self.hypers.model_params['share_cell_weights']:
                # Always include the RNN cell weights
                cell_vars = [VariableWithWeight(var, 1.0) for var in trainable_vars if 'rnn-cell' in var.name]
                layer_vars.extend(cell_vars) 

            loss_ops[f'loss_{i}'] = layer_vars

            # Update the layer weights
            for prefix in var_prefixes:
                var_prefixes[prefix] = var_prefixes[prefix] * layer_weight

        self._loss_ops = loss_ops

    def make_loss(self):
        losses: List[tf.Tensor] = []
        for loss_op_name in self._loss_ops:
            losses.append(self._ops[loss_op_name])
        
        losses = tf.stack(losses)  # B x N, N is the number of sequences
        loss_weights = tf.expand_dims(self._placeholders['loss_weights'], axis=0)
        weighted_losses = tf.reduce_sum(losses * loss_weights, axis=-1)  # B

        self._ops['loss'] = tf.reduce_mean(weighted_losses)  # Weighted average over all layers
 
    def __dynamic_rnn(self,
                      inputs: tf.Tensor,
                      cell: tf.nn.rnn_cell.BasicRNNCell,
                      previous_states: Optional[tf.TensorArray] = None,
                      name: str = None) -> Tuple[tf.TensorArray, tf.TensorArray]:
        
        seq_length = self.metadata['seq_length']
        sample_frac = self.hypers.model_params['sample_frac']
        state_size = self.hypers.model_params['state_size']
        
        samples_per_seq = int(seq_length * sample_frac)
        states_array = tf.TensorArray(dtype=tf.float32, size=samples_per_seq, dynamic_size=False, clear_after_read=False)
        outputs_array = tf.TensorArray(dtype=tf.float32, size=samples_per_seq, dynamic_size=False)

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
            return index < samples_per_seq

        i = tf.constant(0, dtype=tf.int32)
        state = cell.zero_state(batch_size=tf.shape(inputs)[0], dtype=tf.float32)
        _, _, final_outputs, final_states = tf.while_loop(cond=cond,
                                                          body=step,
                                                          loop_vars=[i, state, outputs_array, states_array],
                                                          parallel_iterations=1,
                                                          maximum_iterations=samples_per_seq,
                                                          name='rnn-while-loop')
        return final_outputs, final_states
