import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any

from models.base_model import Model
from dataset.dataset import Dataset, DataSeries
from layers.basic import mlp


class SimpleModel(Model):

    def load_metadata(self, dataset: Dataset):
        input_samples: List[List[int]] = []
        output_samples: List[List[int]] = []

        for sample in dataset.dataset[DataSeries.TRAIN]:
            input_samples.append([sample['input']])
            output_samples.append([sample['output']])

        input_scaler = StandardScaler()
        input_scaler.fit(input_samples)

        output_scaler = StandardScaler()
        output_scaler.fit(output_samples)

        self.metadata['input_scaler'] = input_scaler
        self.metadata['output_scaler'] = output_scaler

    def batch_to_feed_dict(self, batch: Dict[str, List[Any]], is_train: bool) -> Dict[tf.Tensor, np.ndarray]:
        return {
                self._placeholders['input_ph']: np.array(batch['input']).reshape(-1, 1),
                self._placeholders['output_ph']: np.array(batch['output']).reshape(-1, 1),
                self._placeholders['dropout_keep_rate']: self.hypers.model_params.get('dropout_keep_rate') if not is_train else 1.0
        }

    def make_placeholders(self):
        self._placeholders['input_ph'] = tf.placeholder(shape=[None, 1],
                                                        dtype=tf.float32,
                                                        name='input-ph')
        self._placeholders['output_ph'] = tf.placeholder(shape=[None, 1],
                                                         dtype=tf.float32,
                                                         name='output-ph')
        self._placeholders['dropout_keep_rate'] = tf.placeholder(shape=[],
                                                                 dtype=tf.float32,
                                                                 name='dropout-keep-rate')

    def make_model(self):
        with tf.variable_scope('simple_model'):
            predictions = mlp(inputs=self._placeholders['input_ph'],
                              output_size=1,
                              hidden_sizes=[16],
                              activations='relu',
                              dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                              name='model',
                              should_activate_final=False,
                              should_bias_final=False)
            self._ops['output'] = predictions  # B x 1

    def make_loss(self):
        squared_difference = tf.square(self._ops['output'] - self._placeholders['output_ph'])
        mean_squared_error = tf.reduce_mean(squared_difference)
        self._ops['loss'] = mean_squared_error
