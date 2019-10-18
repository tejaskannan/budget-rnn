import tensorflow as tf
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler

from models.base_model import Model
from layers.basic import rnn_cell
from dataset.dataset import Dataset, DataSeries


class RNNSampleModel(Model):

    def load_metadata(self, dataset: Dataset):
        input_samples: List[List[float]] = []
        output_samples: List[List[float]] = []

        for sample in dataset.dataset[DataSeries.TRAIN]:
            input_samples.append(sample['input_power'])
            output_samples.append([sample['output_power']])

        num_input_features = len(input_samples[0][0])
        seq_length = len(input_samples[0])
        input_samples = np.reshape(input_samples, newshape=(-1, num_input_features))

        input_scaler = StandardScaler()
        input_scaler.fit(input_samples)

        output_scaler = StandardScaler()
        output_scaler.fit(output_samples)

        self.metadata['input_scaler'] = input_scaler
        self.metadata['output_scaler'] = output_scaler
        self.metadata['num_input_features'] = num_input_features
        self.metadata['seq_length'] = seq_length

    def batch_to_feed_dict(self, batch: Dict[str, List[Any]], is_train: bool) -> Dict[tf.Tensor, np.ndarray]:
        dropout = self.hypers.model_params.get('dropout_keep_rate', 1.0) if is_train else 1.0
        input_batch = np.array(batch['input_power'])
        output_batch = np.array(batch['output_power'])
        
        input_batch = np.squeeze(input_batch, axis=1)

        feed_dict = {
            self._placeholders['output']: output_batch.reshape(-1, 1),
            self._placeholders['dropout_keep_rate']: dropout
        }

        # Create subsequences
        seq_length = self.metadata['seq_length']
        sample_frac = self.hypers.model_params['sample_frac']
        samples_per_seq = int(seq_length * sample_frac)
        num_sequences = int(1.0 / sample_frac)

        for i in range(num_sequences):
            sample_tensor = np.array([input_batch[:, j] for j in range(i, input_batch.shape[1], num_sequences)])
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
        for i in range(num_sequences):
            # B x S x D
            self._placeholders[f'input_{i}'] = tf.placeholder(shape=[None, samples_per_seq, num_features],
                                                              dtype=tf.float32,
                                                              name=f'input-{i}')

        num_output_features = self.hypers.model_params['num_output_features']
        self._placeholders['output'] = tf.placeholder(shape=[None, num_output_features],
                                                      dtype=tf.float32,
                                                      name='output')  # B x K
        self._placeholders['dropout_keep_rate'] = tf.placeholder(shape=[], dtype=tf.float32, name='dropout-keep-rate')

    def make_model(self, is_train: bool):
        # Create the RNN Cell
        cell = rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                        num_units=self.hypers.model_params['state_size'],
                        activation='tanh',
                        dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                        name='rnn-cell',
                        dtype=tf.float32)

        states_list: List[tf.TensorArray] = []
        outputs_list: List[tf.TensorArray] = []
        num_sequences = int(1.0 / self.hypers.model_params['sample_frac'])
        for i in range(num_sequences):

            with tf.variable_scope(f'layer-{i}'):
                # Create the RNN Cell
                cell = rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                num_units=self.hypers.model_params['state_size'],
                                activation='tanh',
                                dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                                name='rnn-cell',
                                dtype=tf.float32)

                prev_states = states_list[i-1] if i > 0 else None
                rnn_outputs, states = self.__dynamic_rnn(inputs=self._placeholders[f'input_{i}'],
                                                         cell=cell,
                                                         previous_states=prev_states)

                states_list.append(states)
                outputs_list.append(rnn_outputs)

        last_index = tf.shape(self._placeholders[f'input_0'])[1] - 1
        final_states = tf.concat([arr.read(last_index) for arr in states_list], axis=0)  # B * N x D

        output_size = self.hypers.model_params['num_output_features']
        outputs = tf.layers.dense(inputs=final_states,
                                  units=output_size,
                                  activation=None,
                                  use_bias=False,
                                  kernel_initializer=tf.initializers.glorot_uniform(),
                                  name='output-layer')
        reshaped_outputs = tf.reshape(outputs, shape=[-1, num_sequences, output_size])
        self._ops['predictions'] = reshaped_outputs  # B x N x D'

    def make_loss(self):
        # B * N x D'
        num_sequences = int(1.0 / self.hypers.model_params['sample_frac'])
        reshaped_predictions = tf.reshape(self._ops['predictions'], shape=[-1, tf.shape(self._ops['predictions'])[2]])
        tiled_outputs = tf.tile(tf.expand_dims(self._placeholders['output'], axis=1), multiples=(1, num_sequences, 1))
        reshaped_outputs = tf.reshape(tiled_outputs, shape=(-1, tf.shape(tiled_outputs)[2]))

        squared_error = tf.reduce_mean(tf.square(reshaped_outputs - reshaped_predictions), axis=-1)
        mse = tf.reduce_sum(squared_error)
        self._ops['loss'] = mse

    def __dynamic_rnn(self,
                      inputs: tf.Tensor,
                      cell: tf.nn.rnn_cell.BasicRNNCell,
                      previous_states: Optional[tf.TensorArray] = None) -> Tuple[tf.TensorArray, tf.TensorArray]:
        
        seq_length = self.metadata['seq_length']
        sample_frac = self.hypers.model_params['sample_frac']
        state_size = self.hypers.model_params['state_size']
        
        samples_per_seq = int(seq_length * sample_frac)
        states_array = tf.TensorArray(dtype=tf.float32, size=samples_per_seq, dynamic_size=False, clear_after_read=False)
        outputs_array = tf.TensorArray(dtype=tf.float32, size=samples_per_seq, dynamic_size=False)

        if previous_states is not None:
            combine_states = tf.layers.Dense(units=1,
                                             activation=tf.math.sigmoid,
                                             use_bias=True,
                                             kernel_initializer=tf.initializers.glorot_uniform(),
                                             name='combine-states')

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
