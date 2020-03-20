import tensorflow as tf

import numpy as np
import re
import time
from collections import namedtuple, defaultdict, OrderedDict
from typing import List, Optional, Tuple, Dict, Any, Set, Union, DefaultDict, Iterable
from sklearn.preprocessing import StandardScaler

from models.base_model import Model
from layers.basic import rnn_cell, mlp
from layers.cells.cells import make_rnn_cell, MultiRNNCell
from layers.rnn import dynamic_rnn, dropped_rnn, RnnOutput
from layers.output_layers import OutputType, compute_binary_classification_output, compute_regression_output
from layers.embedding_layer import embedding_layer
from dataset.dataset import Dataset, DataSeries
from utils.hyperparameters import HyperParameters
from utils.tfutils import pool_rnn_outputs
from utils.constants import SMALL_NUMBER, BIG_NUMBER, ACCURACY, ONE_HALF, OUTPUT, INPUTS, LOSS
from utils.rnn_utils import *
from utils.testing_utils import ClassificationMetric, RegressionMetric, get_classification_metric, get_regression_metric
from utils.np_utils import thresholded_predictions, sigmoid


class RNNModel(Model):

    def __init__(self, hyper_parameters: HyperParameters, save_folder: str):
        super().__init__(hyper_parameters, save_folder)

        model_type = self.hypers.model_params['model_type'].upper()
        self.model_type = RNNModelType[model_type]

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
    def f1_score_op_names(self) -> List[str]:
        return [get_f1_score_name(i) for i in range(self.num_outputs)]

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
        if self.model_type == RNNModelType.VANILLA and not self.hypers.model_params['share_cell_weights']:
            return [get_loss_name(i) for i in range(self.num_outputs)]
        return ['loss']

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
        input_batch = np.array(batch['inputs'])
        output_batch = np.array(batch['output'])

        if input_batch.shape[1] == 1:
            input_batch = np.squeeze(input_batch, axis=1)

        input_shape = self.metadata['input_shape']
        num_output_features = self.metadata['num_output_features']

        feed_dict = {
            self._placeholders[OUTPUT]: output_batch.reshape(-1, num_output_features),
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

            if self.model_type == RNNModelType.VANILLA:
                seq_indexes.extend(range(i, seq_length, num_sequences))
                seq_indexes = list(sorted(seq_indexes))
                sample_tensor = input_batch[:, seq_indexes]
                feed_dict[input_ph] = sample_tensor
            elif self.model_type in (RNNModelType.SAMPLE, RNNModelType.LINKED):
                seq_indexes = list(range(i, seq_length, num_sequences))
                sample_tensor = input_batch[:, seq_indexes]
                feed_dict[input_ph] = sample_tensor
            else:  # Cascade
                start, end = i * samples_per_seq, (i+1) * samples_per_seq
                sample_tensor = input_batch[:, start:end]
                feed_dict[input_ph] = sample_tensor

        return feed_dict

    def make_placeholders(self):
        """
        Create model placeholders.
        """
        # Extract parameters
        input_features_shape = self.metadata['input_shape']
        num_output_features = self.metadata['num_output_features']
        seq_length = self.metadata['seq_length']
        samples_per_seq = self.samples_per_seq
        num_sequences = self.num_sequences

        # Make input placeholders
        for i in range(num_sequences):
            input_shape = [None, samples_per_seq] + list(input_features_shape)

            # [B, S, D]
            self._placeholders[get_input_name(i)] = tf.placeholder(shape=input_shape,
                                                                   dtype=tf.float32,
                                                                   name=get_input_name(i))
            if self.model_type == RNNModelType.VANILLA:
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

    def predict(self, dataset: Dataset,
                test_batch_size: Optional[int],
                max_num_batches: Optional[int]) -> DefaultDict[str, Dict[str, List[float]]]:
        test_batch_size = test_batch_size if test_batch_size is not None else self.hypers.batch_size
        test_batch_generator = dataset.minibatch_generator(series=DataSeries.TEST,
                                                           batch_size=test_batch_size,
                                                           metadata=self.metadata,
                                                           should_shuffle=False,
                                                           drop_incomplete_batches=True)

        if self.output_type == OutputType.CLASSIFICATION:
            return self.predict_classification(test_batch_generator, test_batch_size, max_num_batches)
        else:  # Regression
            return self.predict_regression(test_batch_generator, test_batch_size, max_num_batches)

    def predict_classification(self, test_batch_generator: Iterable[Any],
                               batch_size: int,
                               max_num_batches: Optional[int]) -> DefaultDict[str, Dict[str, float]]:
        predictions_dict = defaultdict(list)
        latencies_dict = defaultdict(list)
        levels_dict = defaultdict(list)
        labels: List[np.ndarray] = []

        for batch_num, batch in enumerate(test_batch_generator):
            feed_dict = self.batch_to_feed_dict(batch, is_train=False)

            # Execute predictions and time results
            latencies: List[float] = []
            logits_list: List[np.ndarray] = []
            level = 0

            start = time.time()
            prediction_generator = self.anytime_generator(feed_dict, self.num_outputs)
            for prediction_op, (prediction, logits) in zip(self.prediction_ops, prediction_generator):
                predictions_dict[prediction_op].append(prediction)  # [B]
                logits_list.append(logits)
                elapsed = time.time() - start

                latencies.append(elapsed)
                latencies_dict[prediction_op].append(elapsed)

                levels_dict[prediction_op].append(level + 1)
                level += 1

            labels.append(np.vstack(batch[OUTPUT]))

            # Scheduled model
            logits = np.concatenate(logits_list, axis=-1)
            predicted_probs = sigmoid(logits)
            thresholds = [ONE_HALF for _ in range(self.num_outputs)]
            level_output = thresholded_predictions(predicted_probs, thresholds)
            level_predictions = level_output.predictions
            computed_levels = level_output.indices

            scheduled_latencies = [latencies[x] for x in computed_levels]

            predictions_dict['scheduled_model'].append(np.expand_dims(level_predictions, axis=-1))
            latencies_dict['scheduled_model'].append(np.average(scheduled_latencies))
            levels_dict['scheduled_model'].append(np.average(computed_levels + 1.0))

            # Compute metrics for the scheduled model
            if max_num_batches is not None and batch_num >= max_num_batches:
                break

        # Stack all labels into a single array
        labels = np.vstack(labels)

        result = defaultdict(dict)
        for model_name in predictions_dict.keys():

            predictions = np.vstack(predictions_dict[model_name])
            latency = float(np.average(latencies_dict[model_name][1:]))
            levels = float(np.average(levels_dict[model_name]))

            for metric_name in ClassificationMetric:
                metric_value = get_classification_metric(metric_name, predictions, labels, latency, levels)
                result[model_name][metric_name.name] = metric_value

        return result

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
            f1_score_name = get_f1_score_name(i)

            # Create the embedding layer
            input_sequence = embedding_layer(inputs=self._placeholders[input_name],
                                             units=self.hypers.model_params['state_size'],
                                             dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                                             use_conv=self.hypers.model_params['use_conv_embedding'],
                                             params=self.hypers.model_params['embedding_layer_params'],
                                             seq_length=self.samples_per_seq,
                                             input_shape=self.metadata['input_shape'],
                                             name_prefix=f'embedding-{i}')

            # Create the RNN Cell
            cell = make_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                 input_units=self.hypers.model_params['state_size'],
                                 output_units=self.hypers.model_params['state_size'],
                                 activation=self.hypers.model_params['rnn_activation'],
                                 dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                                 num_layers=self.hypers.model_params['rnn_layers'],
                                 name=cell_name)

            inputs = input_sequence
            initial_state = cell.zero_state(batch_size=tf.shape(inputs)[0], dtype=tf.float32)

            # Set the initial state for chunked model types
            if prev_state is not None:
                if self.model_type == RNNModelType.CASCADE or self.hypers.model_params['link_levels']:
                    initial_state = prev_state

            # Set previous states for the Sample model type
            prev_states = None
            if self.model_type == RNNModelType.SAMPLE and i > 0:
                prev_states = states_list[i-1]

            # Run RNN and collect outputs
            rnn_out = dynamic_rnn(cell=cell,
                                  inputs=inputs,
                                  previous_states=prev_states,
                                  initial_state=initial_state,
                                  name=rnn_level_name,
                                  fusion_mode=self.hypers.model_params.get('fusion_mode'))
            rnn_outputs = rnn_out.outputs
            rnn_states = rnn_out.states
            rnn_gates = rnn_out.gates

            # Save previous states
            states_list.append(rnn_states)

            # Get the final state
            last_index = tf.shape(inputs)[1] - 1
            final_output = rnn_outputs.read(index=last_index)
            final_state = tf.squeeze(rnn_states.read(index=last_index), axis=0)  # [B, D]

            # [B, D]
            rnn_output = pool_rnn_outputs(rnn_outputs, final_state, pool_mode=self.hypers.model_params['pool_mode'])

            # [B, K]
            output = mlp(inputs=rnn_output,
                         output_size=num_output_features,
                         hidden_sizes=self.hypers.model_params.get('output_hidden_units'),
                         activations=self.hypers.model_params['output_hidden_activation'],
                         dropout_keep_rate=self._placeholders['dropout_keep_rate'],
                         name=output_layer_name)

            if self.output_type == OutputType.CLASSIFICATION:
                classification_output = compute_binary_classification_output(model_output=output,
                                                                             labels=self._placeholders[OUTPUT],
                                                                             false_pos_weight=self.hypers.model_params['pos_weights'][i],
                                                                             false_neg_weight=self.hypers.model_params['neg_weights'][i],
                                                                             mode=self.hypers.model_params['loss_mode'])

                self._ops[logits_name] = classification_output.logits
                self._ops[prediction_name] = classification_output.predictions
                self._ops[loss_name] = classification_output.loss
                self._ops[accuracy_name] = classification_output.accuracy
                self._ops[f1_score_name] = classification_output.f1_score
            else:
                regression_output = compute_regression_output(model_output=output, expected_otuput=self._placeholders['output'])
                self._ops[prediction_name] = regression_output.predictions
                self._ops[loss_name] = regression_output.loss

            self._ops[gate_name] = rnn_gates.stack()  # [B, T, M, D]
            self._ops[state_name] = rnn_states.stack()  # [B, T, D]

            # Save previous state for possible reuse at the next level
            prev_state = rnn_states.read(index=last_index)

            outputs.append(output)

        combined_outputs = tf.concat(tf.nest.map_structure(lambda t: tf.expand_dims(t, axis=1), outputs), axis=1)
        self._ops[ALL_PREDICTIONS_NAME] = combined_outputs

    def make_loss(self):
        losses: List[tf.Tensor] = []

        # The loss_op keys are ordered by the output level
        for level in range(self.num_outputs):
            batch_loss = self._ops[get_loss_name(level)]
            losses.append(tf.reduce_mean(batch_loss))

        losses = tf.stack(losses)  # [N], N is the number of sequences
        weighted_losses = tf.reduce_sum(losses * self._placeholders['loss_weights'], axis=-1)  # Scalar

        self._ops[LOSS] = weighted_losses

    def anytime_generator(self, feed_dict: Dict[tf.Tensor, List[Any]],
                          max_num_levels: int) -> Optional[Iterable[np.ndarray]]:
        """
        Anytime Inference in a generator-like fashion using Tensorflow's partial run API
        """
        with self.sess.graph.as_default():
            # Initialize partial run settings
            prediction_ops = [self._ops[op_name] for op_name in self.prediction_ops]
            logit_ops = [self._ops[get_logits_name(i)] for i in range(self.num_outputs)]
            ops_to_run = prediction_ops + logit_ops

            placeholders = list(self._placeholders.values())
            handle = self.sess.partial_run_setup(ops_to_run, placeholders)

            result = None
            for level in range(max_num_levels):
                prediction_op = prediction_ops[level]
                logit_op = logit_ops[level]

                # Filter feed dict to avoid feeding the same inputs multiple times
                op_feed_dict = {key: value for key, value in feed_dict.items() if key.name.endswith(str(level))}
                prediction, logits = self.sess.partial_run(handle, [prediction_op, logit_op], feed_dict=op_feed_dict)

                yield prediction, logits
