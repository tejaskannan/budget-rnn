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
from layers.output_layers import OutputType, compute_binary_classification_output
from layers.embedding_layer import embedding_layer
from dataset.dataset import Dataset, DataSeries
from utils.hyperparameters import HyperParameters
from utils.tfutils import pool_rnn_outputs
from utils.constants import SMALL_NUMBER, BIG_NUMBER, ACCURACY, ONE_HALF, OUTPUT, INPUTS, LOSS
from utils.constants import NODE_REGEX_FORMAT, DROPOUT_KEEP_RATE, MODEL, SCHEDULED_MODEL
from utils.constants import INPUT_SCALER, OUTPUT_SCALER, INPUT_SHAPE, NUM_OUTPUT_FEATURES, SEQ_LENGTH
from utils.loss_utils import f1_score_loss, binary_classification_loss
from utils.rnn_utils import *
from utils.testing_utils import ClassificationMetric, RegressionMetric, get_classification_metric, get_regression_metric, ALL_LATENCY
from utils.np_utils import sigmoid
from utils.threshold_utils import lower_threshold_predictions, TwoSidedThreshold


class AdaptiveModel(Model):

    def __init__(self, hyper_parameters: HyperParameters, save_folder: str, is_train: bool):
        super().__init__(hyper_parameters, save_folder, is_train)

        model_type = self.hypers.model_params['model_type'].upper()
        self.model_type = RNNModelType[model_type]

        self.name = model_type

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
    def logit_op_names(self) -> List[str]:
        return [get_logits_name(i) for i in range(self.num_outputs)]

    @property
    def output_ops(self) -> List[str]:
        return self.prediction_ops

    @property
    def samples_per_seq(self) -> int:
        seq_length = self.metadata['seq_length']
        return int(seq_length * self.sample_frac)

    @property
    def loss_op_names(self) -> List[str]:
        if self.model_type == RNNModelType.VANILLA and not self.hypers.model_params['share_cell_weights']:
            return [get_loss_name(i) for i in range(self.num_outputs)]
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

        self.metadata[INPUT_SCALER] = input_scaler
        self.metadata[OUTPUT_SCALER] = output_scaler
        self.metadata[INPUT_SHAPE] = input_shape
        self.metadata[NUM_OUTPUT_FEATURES] = num_output_features
        self.metadata[SEQ_LENGTH] = seq_length

    def batch_to_feed_dict(self, batch: Dict[str, List[Any]], is_train: bool) -> Dict[tf.Tensor, np.ndarray]:
        dropout = self.hypers.dropout_keep_rate if is_train else 1.0
        input_batch = np.array(batch[INPUTS])
        output_batch = np.array(batch[OUTPUT])

        if input_batch.shape[1] == 1:
            input_batch = np.squeeze(input_batch, axis=1)

        input_shape = self.metadata[INPUT_SHAPE]
        num_output_features = self.metadata['num_output_features']

        feed_dict = {
            self._placeholders[OUTPUT]: output_batch.reshape(-1, num_output_features),
            self._placeholders[DROPOUT_KEEP_RATE]: dropout
        }

        # Extract parameters
        seq_length = self.metadata[SEQ_LENGTH]
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
        feed_dict[self._placeholders['loss_weights']] = loss_weights / (np.sum(loss_weights) + SMALL_NUMBER)  # Normalize the loss weights

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

    def make_placeholders(self, is_frozen: bool = False):
        """
        Create model placeholders.
        """
        # Extract parameters
        input_features_shape = self.metadata[INPUT_SHAPE]
        num_output_features = self.metadata[NUM_OUTPUT_FEATURES]
        seq_length = self.metadata[SEQ_LENGTH]
        samples_per_seq = self.samples_per_seq
        num_sequences = self.num_sequences

        # Make input placeholders
        for i in range(num_sequences):
            input_shape = [None, samples_per_seq] + list(input_features_shape)

            # [B, S, D]
            if not is_frozen:
                self._placeholders[get_input_name(i)] = tf.placeholder(shape=input_shape,
                                                                       dtype=tf.float32,
                                                                       name=get_input_name(i))
            else:
                self._placeholders[get_input_name(i)] = tf.ones(shape=[1] + input_shape[1:],
                                                                dtype=tf.float32,
                                                                name=get_input_name(i))

            if self.model_type == RNNModelType.VANILLA:
                samples_per_seq += self.samples_per_seq

        if not is_frozen:
            # [B, K]
            self._placeholders[OUTPUT] = tf.placeholder(shape=[None, num_output_features],
                                                        dtype=tf.float32,
                                                        name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = tf.placeholder(shape=[],
                                                                   dtype=tf.float32,
                                                                   name=DROPOUT_KEEP_RATE)
            self._placeholders['loss_weights'] = tf.placeholder(shape=[self.num_outputs],
                                                                dtype=tf.float32,
                                                                name='loss-weights')
        else:
            self._placeholders[OUTPUT] = tf.ones(shape=[1, num_output_features], dtype=tf.float32, name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = tf.ones(shape=[], dtype=tf.float32, name=DROPOUT_KEEP_RATE)
            self._placeholders['loss_weights'] = tf.ones(shape=[self.num_outputs], dtype=tf.float32, name='loss-weights')

    def compute_flops(self, level: int) -> int:
        """
        Computes the total floating point operations for the given prediction level
        """
        if level < 0:
            return 0

        total_flops = 0

        rm = tf.RunMetadata()
        with self.sess.graph.as_default():
            cell = get_cell_level_name(level, self.hypers.model_params['share_cell_weights'])
            output = get_output_layer_name(level)
            rnn = get_rnn_level_name(level)
            embedding = get_embedding_name(level)
            combine_states = get_combine_states_name(rnn)

            # Compute FLOPS from RNN operations
            rnn_operations = list(map(lambda t: NODE_REGEX_FORMAT.format(t), [cell, rnn, combine_states]))
            rnn_options = tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.float_operation()) \
                                .with_node_names(show_name_regexes=rnn_operations) \
                                .order_by('flops').build()
            flops = tf.profiler.profile(self.sess.graph, options=rnn_options)

            flops_factor = level + 1 if self.model_type == RNNModelType.VANILLA else 1
            total_flops += flops.total_float_ops * flops_factor * self.samples_per_seq

            # Compute FLOPS from all other operations. Note that the embedding layer has a well-defined input length,
            # so Tensorflow already accounts for the repeated application of this transformation.
            single_operations = list(map(lambda t: NODE_REGEX_FORMAT.format(t), [output, embedding]))
            single_options = tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.float_operation()) \
                                .with_node_names(show_name_regexes=single_operations) \
                                .order_by('flops').build()
            flops = tf.profiler.profile(self.sess.graph, options=single_options)
            total_flops += flops.total_float_ops

        return total_flops


    def predict_classification(self, test_batch_generator: Iterable[Any],
                               batch_size: int,
                               max_num_batches: Optional[int],
                               flops_dict: Optional[Dict[str, int]]) -> DefaultDict[str, Dict[str, Any]]:
        predictions_dict = defaultdict(list)
        latencies_dict = defaultdict(list)
        levels_dict = defaultdict(list)
        labels: List[np.ndarray] = []

        # Standard, baseline thresholds
        thresholds = [TwoSidedThreshold(lower=ONE_HALF, upper=1.0) for _ in range(self.num_outputs)]

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

                start = time.time()

            labels.append(np.vstack(batch[OUTPUT]))

            # Scheduled model
            logits = np.concatenate(logits_list, axis=-1)
            predicted_probs = sigmoid(logits)
            level_output = lower_threshold_predictions(predicted_probs, thresholds)
            level_predictions = level_output.predictions
            computed_levels = level_output.indices

            scheduled_latencies = [latencies[x] for x in computed_levels]
            scheduled_flops = [flops_dict[get_prediction_name(x)] for x in computed_levels]

            predictions_dict[SCHEDULED_MODEL].append(np.expand_dims(level_predictions, axis=-1))
            latencies_dict[SCHEDULED_MODEL].append(np.average(scheduled_latencies))
            levels_dict[SCHEDULED_MODEL].append(np.average(computed_levels + 1.0))
            flops_dict[SCHEDULED_MODEL] = np.average(scheduled_flops)

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
            flops = flops_dict[model_name]

            for metric_name in ClassificationMetric:
                metric_value = get_classification_metric(metric_name, predictions, labels, latency, levels, flops)
                result[model_name][metric_name.name] = metric_value

            result[model_name][ALL_LATENCY] = latencies_dict[model_name][1:]

        return result

    def make_model(self, is_train: bool):
        with tf.variable_scope(MODEL, reuse=tf.AUTO_REUSE):
            self.make_rnn_model(is_train)

    def make_rnn_model(self, is_train: bool):
        outputs: List[tf.Tensor] = []
        states_list: List[tf.TensorArray] = []
        prev_state: Optional[tf.Tensor] = None

        num_output_features = self.metadata[NUM_OUTPUT_FEATURES]

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
                                             dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                             use_conv=self.hypers.model_params['use_conv_embedding'],
                                             params=self.hypers.model_params['embedding_layer_params'],
                                             seq_length=self.samples_per_seq,
                                             input_shape=self.metadata[INPUT_SHAPE],
                                             name_prefix=get_embedding_name(i))

            # Create the RNN Cell
            cell = make_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                 input_units=self.hypers.model_params['state_size'],
                                 output_units=self.hypers.model_params['state_size'],
                                 activation=self.hypers.model_params['rnn_activation'],
                                 dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
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
                         dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                         name=output_layer_name)

            if self.output_type == OutputType.CLASSIFICATION:
                classification_output = compute_binary_classification_output(model_output=output,
                                                                             labels=self._placeholders[OUTPUT])

                self._ops[logits_name] = classification_output.logits
                self._ops[prediction_name] = classification_output.predictions
                self._ops[accuracy_name] = classification_output.accuracy
                self._ops[f1_score_name] = classification_output.f1_score
            else:
                self._ops[prediction_name] = output

            self._ops[gate_name] = rnn_gates.stack()  # [B, T, M, D]
            self._ops[state_name] = rnn_states.stack()  # [B, T, D]

            # Save previous state for possible reuse at the next level
            prev_state = rnn_states.read(index=last_index)

            outputs.append(output)

        combined_outputs = tf.concat(tf.nest.map_structure(lambda t: tf.expand_dims(t, axis=1), outputs), axis=1)
        self._ops[ALL_PREDICTIONS_NAME] = combined_outputs

    def make_loss(self):
        losses: List[tf.Tensor] = []

        loss_mode = self.hypers.model_params['loss_mode'].lower()

        # The loss_op keys are ordered by the output level
        for level in range(self.num_outputs):
            loss_name = get_loss_name(level)
            expected_output = self._placeholders[OUTPUT]
            predictions = self._ops[get_prediction_name(level)]

            if self.output_type == OutputType.CLASSIFICATION:
                logits = self.ops[get_logits_name(level)]
                predicted_probs = tf.math.sigmoid(logits)

                if loss_mode in ('cross-entropy', 'cross-entropy', 'accuracy'):
                    self._ops[loss_name] = binary_classification_loss(predicted_probs=predicted_probs,
                                                                      predictions=predictions,
                                                                      labels=expected_output,
                                                                      pos_weight=self.hypers.model_params['pos_weights'][level],
                                                                      neg_weight=self.hypers.model_params['neg_weights'][level])
                elif loss_mode in ('f1', 'f1_score', 'f1-score'):
                    self._ops[loss_name] = f1_score_loss(predicted_probs=predicted_probs, labels=expected_output)
                else:
                    raise ValueError(f'Unknown loss mode: {loss_mode}')
            else:  # Regression task
                self._ops[loss_name] = tf.reduce_mean(tf.square(predictions - expected_output))  # MSE loss

            batch_loss = self._ops[loss_name]
            losses.append(tf.reduce_mean(batch_loss))

        losses = tf.stack(losses)  # [N], N is the number of sequences
        weighted_losses = tf.reduce_sum(losses * self._placeholders['loss_weights'], axis=-1)  # Scalar

        self._ops[LOSS] = weighted_losses

    def anytime_generator(self, feed_dict: Dict[tf.Tensor, List[Any]],
                          max_num_levels: int) -> Optional[Iterable[np.ndarray]]:
        """
        Anytime Inference in a generator-like fashion
        """
        with self.sess.graph.as_default():
            num_levels = max(self.num_outputs, max_num_levels) if max_num_levels is not None else self.num_outputs

            for level in range(num_levels):
                prediction_ops = [get_prediction_name(i) for i in range(level + 1)]
                logit_ops = [get_logits_name(i) for i in range(level + 1)]

                ops_to_run = prediction_ops + logit_ops
                output_dict = self.execute(feed_dict, ops_to_run)

                prediction_name = get_prediction_name(level)
                logits_name = get_logits_name(level)

                yield output_dict[prediction_name], output_dict[logits_name]
