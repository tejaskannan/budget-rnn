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
        dropout = self.hypers.model_params.get('dropout_keep_rate', 1.0) if is_train else 1.0
        input_batch = np.array(batch['input'])
        output_batch = np.array(batch['output'])

        return {
                self._placeholders['input_ph']: input_batch,
                self._placeholders['output_ph']: output_batch,
                self._placeholders['dropout_keep_rate']: dropout
        }

    def make_placeholders(self):
        self._placeholders['input_ph'] = tf.placeholder(shape=[None, 1, 1],
                                                        dtype=tf.float32,
                                                        name='input-ph')
        self._placeholders['output_ph'] = tf.placeholder(shape=[None, 1, 1],
                                                         dtype=tf.float32,
                                                         name='output-ph')
        self._placeholders['dropout_keep_rate'] = tf.placeholder(shape=[],
                                                                 dtype=tf.float32,
                                                                 name='dropout-keep-rate')

    def make_model(self):
        with tf.variable_scope('simple_model'):
            predictions = mlp(inputs=self._placeholders['input_ph'],
                              output_size=1,
                              hidden_sizes=[4],
                              activations=None,
                              dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                              name='model',
                              should_activate_final=False,
                              should_bias_final=False,
                              should_dropout_final=False)
            self._ops['output'] = predictions  # B x 1

    def predict(self, samples: List[Dict[str, Any]], dataset: Dataset) -> Dict[Any, Any]:
        batch: Dict[str, List[Any]] = {'input': [], 'output': []}
        for sample in samples:
            tensorized_sample = dataset.tensorize(sample, self.metadata)
            for key, tensor in tensorized_sample.items():
                batch[key].append(tensor)

        feed_dict = self.batch_to_feed_dict(batch, is_train=False)
        op_result = self.execute(feed_dict, 'output')

        print(batch)

        original_inputs = self.metadata['input_scaler'].inverse_transform(np.array(batch['input']).reshape(-1, 1))
        outputs = self.metadata['output_scaler'].inverse_transform(op_result['output'].reshape(-1, 1))

        return {input_val[0]: output_val[0] for input_val, output_val in zip(original_inputs, outputs)}

    def make_loss(self):
        squared_difference = tf.square(self._ops['output'] - self._placeholders['output_ph'])
        mean_squared_error = tf.reduce_mean(squared_difference)
        self._ops['loss'] = mean_squared_error
