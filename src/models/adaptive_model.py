import tensorflow as tf
import numpy as np
import re
import time
from collections import namedtuple, defaultdict, OrderedDict
from typing import List, Optional, Tuple, Dict, Any, Set, Union, DefaultDict, Iterable

from models.tf_model import TFModel
from layers.basic import rnn_cell, mlp, dense
from layers.cells.cells import make_rnn_cell, MultiRNNCell
from layers.rnn import dynamic_rnn, RnnOutput
from layers.output_layers import OutputType, compute_binary_classification_output, compute_multi_classification_output
from dataset.dataset import Dataset, DataSeries
from utils.hyperparameters import HyperParameters
from utils.tfutils import pool_rnn_outputs, expand_to_matrix
from utils.misc import sample_sequence_batch, batch_sample_noise
from utils.constants import SMALL_NUMBER, BIG_NUMBER, ACCURACY, OUTPUT, INPUTS, LOSS, OUTPUT_SEED, OPTIMIZER_OP, GLOBAL_STEP
from utils.constants import NODE_REGEX_FORMAT, DROPOUT_KEEP_RATE, MODEL, SCHEDULED_MODEL, NUM_CLASSES, TRANSFORM_SEED
from utils.constants import INPUT_SHAPE, NUM_OUTPUT_FEATURES, SEQ_LENGTH, INPUT_NOISE, EMBEDDING_SEED, AGGREGATE_SEED, STOP_LOSS_WEIGHT
from utils.loss_utils import f1_score_loss, binary_classification_loss, get_loss_weights
from utils.rnn_utils import *
from utils.testing_utils import ClassificationMetric, RegressionMetric, get_binary_classification_metric, get_regression_metric, ALL_LATENCY, get_multi_classification_metric
from utils.np_utils import sigmoid


class AdaptiveModel(TFModel):

    def __init__(self, hyper_parameters: HyperParameters, save_folder: str, is_train: bool):
        super().__init__(hyper_parameters, save_folder, is_train)

        model_type = self.hypers.model_params['model_type'].upper()
        self.model_type = AdaptiveModelType[model_type]

        self.name = model_type

        self._has_shared_weights = any(val for key, val in self.hypers.model_params.items() if key.startswith('share') and key.endswith('weights'))

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
        if is_independent(self.model_type) and not self._has_shared_weights:
            return [get_loss_name(i) for i in range(self.num_outputs)]
        return [LOSS]

    @property
    def optimizer_op_names(self) -> List[str]:
        if is_independent(self.model_type) and not self._has_shared_weights:
            return ['{0}_{1}'.format(OPTIMIZER_OP, i) for i in range(self.num_outputs)]
        return [OPTIMIZER_OP]

    @property
    def global_step_op_names(self) -> List[str]:
        if is_independent(self.model_type) and not self._has_shared_weights:
            return ['{0}_{1}'.format(GLOBAL_STEP, i) for i in range(self.num_outputs)]
        return [GLOBAL_STEP]

    def batch_to_feed_dict(self, batch: Dict[str, List[Any]], is_train: bool, epoch_num: int) -> Dict[tf.Tensor, np.ndarray]:
        dropout = self.hypers.dropout_keep_rate if is_train else 1.0
        input_batch = np.array(batch[INPUTS])
        output_batch = np.array(batch[OUTPUT])

        if input_batch.shape[1] == 1:
            input_batch = np.squeeze(input_batch, axis=1)

        input_shape = self.metadata[INPUT_SHAPE]
        num_output_features = self.metadata[NUM_OUTPUT_FEATURES]
        seq_lenth = self.metadata[SEQ_LENGTH]

        # Calculate the stop loss weight based on the patience, number of levels, and epoch number
        # This weight will start at 1 / (L + Patience) and end at 1 / L after `patience` epochs.
        # We clip the output to 1 / L as a maximum value.
        if self.hypers.model_params.get('should_increase_stop_loss_weight', False):
            reciprocal_stop_loss_weight = self.num_sequences + max(self.hypers.patience - epoch_num, 0)
            stop_loss_weight = 1.0 / reciprocal_stop_loss_weight
        else:
            stop_loss_weight = self.hypers.model_params.get('stop_loss_weight', 0.0)

        feed_dict = {
            self._placeholders[OUTPUT]: output_batch.reshape(-1, num_output_features),
            self._placeholders[DROPOUT_KEEP_RATE]: dropout,
            self._placeholders[STOP_LOSS_WEIGHT]: stop_loss_weight
        }

        # Sample the batch down to the correct sequence length
        input_batch = sample_sequence_batch(input_batch, seq_length=self.metadata[SEQ_LENGTH])
        input_batch = batch_sample_noise(input_batch, noise_weight=self.hypers.batch_noise)

        # Extract parameters
        seq_length = self.metadata[SEQ_LENGTH]
        samples_per_seq = self.samples_per_seq
        num_sequences = self.num_sequences
        num_outputs = self.num_outputs

        # The loss weights are sorted in ascending order to more-heavily weight larger sample sizes.
        # This operation is included to prevent bugs as we intuitively want to increase
        # accuracy with larger samples.
        loss_weights = get_loss_weights(n=num_outputs, mode=self.hypers.model_params.get('loss_weights'))
        feed_dict[self._placeholders['loss_weights']] = loss_weights  # Normalize the loss weights

        seq_indexes: List[int] = []
        for i in range(num_sequences):
            input_ph = self.placeholders[get_input_name(i)]

            if is_independent(self.model_type):  # Independent RNN, BIRNN or NBOW
                seq_indexes.extend(range(i, seq_length, num_sequences))
                seq_indexes = list(sorted(seq_indexes))
                sample_tensor = input_batch[:, seq_indexes]
                feed_dict[input_ph] = sample_tensor
            elif is_sample(self.model_type):
                seq_indexes = list(range(i, seq_length, num_sequences))
                sample_tensor = input_batch[:, seq_indexes]
                feed_dict[input_ph] = sample_tensor
            else:  # Cascade RNN or NBOW
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

        output_dtype = tf.int32 if self.output_type == OutputType.MULTI_CLASSIFICATION else tf.float32

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

            if is_independent(self.model_type):
                samples_per_seq += self.samples_per_seq

        if not is_frozen:
            # [B, K]
            self._placeholders[OUTPUT] = tf.placeholder(shape=[None, num_output_features],
                                                        dtype=output_dtype,
                                                        name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = tf.placeholder(shape=[],
                                                                   dtype=tf.float32,
                                                                   name=DROPOUT_KEEP_RATE)
            self._placeholders['loss_weights'] = tf.placeholder(shape=[self.num_outputs],
                                                                dtype=tf.float32,
                                                                name='loss-weights')
            self._placeholders[STOP_LOSS_WEIGHT] = tf.placeholder(shape=[],
                                                                  dtype=tf.float32,
                                                                  name=STOP_LOSS_WEIGHT)
        else:
            self._placeholders[OUTPUT] = tf.ones(shape=[1, num_output_features], dtype=output_dtype, name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = tf.ones(shape=[], dtype=tf.float32, name=DROPOUT_KEEP_RATE)
            self._placeholders[STOP_LOSS_WEIGHT] = tf.ones(shape=[], dtype=tf.float32, name=STOP_LOSS_WEIGHT)
            self._placeholders['loss_weights'] = tf.ones(shape=[self.num_outputs], dtype=tf.float32, name='loss-weights')

    def compute_flops(self, level: int) -> int:
        """
        Computes the total floating point operations for the given prediction level
        """
        if is_nbow(self.model_type):
            return self.compute_nbow_flops(level)
        return self.compute_rnn_flops(level)

    def compute_nbow_flops(self, level: int) -> int:
        if level < 0:
            return 0

        total_flops = 0
        rm = tf.RunMetadata()
        with self.sess.graph.as_default():
            transform_name = get_transform_name(level, self.hypers.model_params['share_transform_weights'])
            aggregation_name = get_aggregation_name(level, self.hypers.model_params['share_transform_weights'])
            embedding_name = get_embedding_name(level, self.hypers.model_params['share_embedding_weights'])
            output_name = get_output_layer_name(level, self.hypers.model_params['share_output_weights'])

            # Compute FLOPS for the transformation layer and aggregation dense layer
            if self.hypers.model_params['share_transform_weights']:
                if level == 0:
                    transform_regex = '^.*{0}([^_]+)$'.format(transform_name)
                    agg_dense_regex = '^.*{0}/.*$'.format(aggregation_name)
                else:
                    transform_regex = '.*{0}.*_{1}.*'.format(transform_name, level)
                    agg_dense_regex = '.*{0}_{1}/.*'.format(aggregation_name, level)
            else:
                transform_regex = NODE_REGEX_FORMAT.format(transform_name)
                agg_dense_regex = '.*{0}-([^0-9]+).*'.format(aggregation_name)

            if self.hypers.model_params['share_output_weights']:
                if level == 0:
                    output_regex = '^.*{0}([^_]+)$'.format(output_name)
                else:
                    output_regex = '.*{0}.*_{1}.*'.format(output_name, level)
            else:
                output_regex = NODE_REGEX_FORMAT.format(output_name)

            agg_ops_regex = '.*{0}-{1}.*'.format(aggregation_name, level)

            op_names = [transform_regex, agg_dense_regex, output_regex, agg_ops_regex]
            options = tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.float_operation()) \
                            .with_node_names(show_name_regexes=op_names) \
                            .order_by('flops').build()
            level_flops = tf.profiler.profile(self.sess.graph, options=options)
            total_flops += level_flops.total_float_ops

            # Get FLOPS for the embedding layer. We do this in a `marginal' way because there is no need to ever re-compute
            # an embedding (they are all independent). Thus, the number of additional embedding computations is equal
            # to the number of operations on the first level.
            embedding_regex = ['.*{0}-dense/.*'.format(embedding_name), '.*{0}-filter-[0-9]+/.*'.format(embedding_name)]
            embedding_options = tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.float_operation()) \
                                    .with_node_names(show_name_regexes=embedding_regex) \
                                    .order_by('flops').build()
            flops = tf.profiler.profile(self.sess.graph, options=embedding_options)
            total_flops += flops.total_float_ops

        return total_flops

    def compute_rnn_flops(self, level: int) -> int:
        if level < 0:
            return 0

        total_flops = 0

        rm = tf.RunMetadata()
        with self.sess.graph.as_default():
            cell = get_cell_level_name(level, self.hypers.model_params['share_cell_weights'])
            output = get_output_layer_name(level, self.hypers.model_params['share_output_weights'])
            rnn = get_rnn_level_name(level)
            embedding_name = get_embedding_name(level, self.hypers.model_params.get('share_embedding_weights', True))
            combine_states = get_combine_states_name(rnn, self.hypers.model_params['share_rnn_weights'])

            # Compute FLOPS from RNN operations
            op_names = [cell, rnn]
            if not self.hypers.model_params['share_rnn_weights'] or level > 0:
                op_names.append(combine_states)

            rnn_operations = list(map(lambda t: NODE_REGEX_FORMAT.format(t), op_names))
            rnn_options = tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.float_operation()) \
                                .with_node_names(show_name_regexes=rnn_operations) \
                                .order_by('flops').build()
            flops = tf.profiler.profile(self.sess.graph, options=rnn_options)

            flops_factor = level + 1 if is_independent(self.model_type) else 1
            total_flops += flops.total_float_ops * flops_factor * self.samples_per_seq

            # Compute FLOPS for the output layer
            if self.hypers.model_params['share_output_weights']:
                if level == 0:
                    output_regex = '^.*{0}([^_]+)$'.format(output)
                else:
                    output_regex = '.*{0}.*_{1}.*'.format(output, level)
            else:
                output_regex = NODE_REGEX_FORMAT.format(output)

            single_options = tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.float_operation()) \
                                .with_node_names(show_name_regexes=[output_regex]) \
                                .order_by('flops').build()
            flops = tf.profiler.profile(self.sess.graph, options=single_options)
            total_flops += flops.total_float_ops

            # Get FLOPS for the embedding layer. We do this in a `marginal' way because there is no need to ever re-compute
            # an embedding (they are all independent). Thus, the number of additional embedding computations is equal
            # to the number of operations on the first level.
            embedding_regex = ['.*{0}-dense/.*'.format(embedding_name), '.*{0}-filter-[0-9]+/.*'.format(embedding_name)]
            embedding_options = tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.float_operation()) \
                                    .with_node_names(show_name_regexes=embedding_regex) \
                                    .order_by('flops').build()
            flops = tf.profiler.profile(self.sess.graph, options=embedding_options)
            total_flops += flops.total_float_ops

        return total_flops

    def predict_regression(self, test_batch_generator: Iterable[Any],
                           batch_size: int,
                           max_num_batches: Optional[int],
                           flops_dict: Optional[Dict[str, int]]) -> DefaultDict[str, Dict[str, Any]]:
        predictions_dict = defaultdict(list)
        latencies_dict = defaultdict(list)
        levels_dict = defaultdict(list)
        outputs: List[np.ndarray] = []

        for batch_num, batch in enumerate(test_batch_generator):
            feed_dict = self.batch_to_feed_dict(batch, is_train=False, epoch_num=0)

            # Execute predictions and time results
            latencies: List[float] = []
            level = 0

            start = time.time()
            prediction_generator = self.anytime_generator(feed_dict, self.num_outputs)
            for prediction_op, (prediction, _) in zip(self.prediction_ops, prediction_generator):
                predictions_dict[prediction_op].append(np.vstack(prediction))  # [B]

                elapsed = time.time() - start

                latencies.append(elapsed)
                latencies_dict[prediction_op].append(elapsed)

                levels_dict[prediction_op].append(level + 1)
                level += 1

                start = time.time()

            outputs.append(np.vstack(batch[OUTPUT]))

            if max_num_batches is not None and batch_num >= max_num_batches:
                break

        outputs = np.vstack(outputs)

        result = defaultdict(dict)
        for model_name in predictions_dict.keys():

            predictions = np.vstack(predictions_dict[model_name])
            latency = float(np.average(latencies_dict[model_name][1:]))
            levels = float(np.average(levels_dict[model_name]))
            flops = flops_dict[model_name]

            for metric_name in RegressionMetric:
                metric_value = get_regression_metric(metric_name, predictions, outputs, latency, levels, flops)
                result[model_name][metric_name.name] = metric_value

            # Remove first latency to remove outliers due to startup costs
            result[model_name][ALL_LATENCY] = latencies_dict[model_name][1:]

        return result

    def predict_classification(self, test_batch_generator: Iterable[Any],
                               batch_size: int,
                               max_num_batches: Optional[int],
                               flops_dict: Optional[Dict[str, int]]) -> DefaultDict[str, Dict[str, Any]]:
        predictions_dict = defaultdict(list)
        latencies_dict = defaultdict(list)
        levels_dict = defaultdict(list)
        labels: List[np.ndarray] = []

        for batch_num, batch in enumerate(test_batch_generator):
            feed_dict = self.batch_to_feed_dict(batch, is_train=False, epoch_num=0)

            # Execute predictions and time results
            latencies: List[float] = []
            logits_list: List[np.ndarray] = []
            level = 0

            start = time.time()
            prediction_generator = self.anytime_generator(feed_dict, self.num_outputs)
            for prediction_op, (prediction, logits) in zip(self.prediction_ops, prediction_generator):
                predictions_dict[prediction_op].append(np.vstack(prediction))  # [B]
                logits_list.append(logits)

                elapsed = time.time() - start

                latencies.append(elapsed)
                latencies_dict[prediction_op].append(elapsed)

                levels_dict[prediction_op].append(level + 1)
                level += 1

                start = time.time()

            labels.append(np.vstack(batch[OUTPUT]))

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
                if self.output_type == OutputType.BINARY_CLASSIFICATION:
                    metric_value = get_binary_classification_metric(metric_name, predictions, labels, latency, levels, flops)
                else:
                    metric_value = get_multi_classification_metric(metric_name, predictions, labels, latency, levels, flops, self.metadata[NUM_CLASSES])

                result[model_name][metric_name.name] = metric_value

            # Remove first latency to remove outliers due to startup costs
            result[model_name][ALL_LATENCY] = latencies_dict[model_name][1:]

        return result

    def make_model(self, is_train: bool):
        with tf.variable_scope(MODEL, reuse=tf.AUTO_REUSE):
            if is_nbow(self.model_type):
                self._make_nbow_model(is_train)
            else:
                self._make_rnn_model(is_train)

    def _make_nbow_model(self, is_train: bool):
        outputs: List[tf.Tensor] = []
        stop_outputs: List[tf.Tensor]= []
        all_attn_weights: List[tf.Tensor] = []  # List of [B, T, 1] tensors
        all_samples: List[tf.Tensor] = []  # List of [B, T, D] tensors
        compression_fraction = self.hypers.model_params.get('compression_fraction')

        # Lists for when pool_outputs is True
        level_outputs: List[tf.Tensor] = []
        output_attn_weights: List[tf.Tensor] = []

        for i in range(self.num_sequences):
            # Get relevant variable names
            input_name = get_input_name(i)
            transform_name = get_transform_name(i, self.hypers.model_params['share_transform_weights'])
            aggregation_name = get_aggregation_name(i, self.hypers.model_params['share_transform_weights'])
            output_layer_name = get_output_layer_name(i, self.hypers.model_params['share_output_weights'])
            stop_output_name = get_stop_output_name(i)
            logits_name = get_logits_name(i)
            prediction_name = get_prediction_name(i)
            loss_name = get_loss_name(i)
            gate_name = get_gates_name(i)
            state_name = get_states_name(i)
            accuracy_name = get_accuracy_name(i)
            f1_score_name = get_f1_score_name(i)
            embedding_name = get_embedding_name(i, self.hypers.model_params['share_embedding_weights'])

            # Create the embedding layer. Output is a [B, T, D] tensor where T is the seq length of this level.
            input_sequence, _ = dense(inputs=self._placeholders[input_name],
                                      units=self.hypers.model_params['state_size'],
                                      activation=self.hypers.model_params['embedding_activation'],
                                      use_bias=True,
                                      name=embedding_name,
                                      compression_seed=EMBEDDING_SEED,
                                      compression_fraction=compression_fraction)

            self._ops['embedding_{0}'.format(i)] = input_sequence

            # Transform the input sequence, [B, T, D]
            transformed_sequence, _ = mlp(inputs=input_sequence,
                                          output_size=self.hypers.model_params['state_size'],
                                          hidden_sizes=self.hypers.model_params['transform_units'],
                                          activations=self.hypers.model_params['transform_activation'],
                                          dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                          should_activate_final=True,
                                          should_bias_final=True,
                                          should_dropout_final=True,
                                          name=transform_name,
                                          compression_seed=TRANSFORM_SEED,
                                          compression_fraction=compression_fraction)
            # Save the states
            self._ops[state_name] = tf.transpose(transformed_sequence, perm=[1, 0, 2])  # [T, B, D]

            self._ops['transformed_{0}'.format(i)] = transformed_sequence

            # Compute attention weights for aggregation. We only compute the
            # weights for this sequence to avoid redundant computation. [B, T, 1] tensor.
            attn_weights, _ = dense(inputs=transformed_sequence,
                                    units=1,
                                    activation=self.hypers.model_params['attn_activation'],
                                    use_bias=True,
                                    name=aggregation_name,
                                    compression_seed=AGGREGATE_SEED,
                                    compression_fraction=compression_fraction)

            self._ops['attn_weights_{0}'.format(i)] = attn_weights

            # Save results of this level to avoid redundant computation
            all_attn_weights.append(attn_weights)  # List of [B, T, 1] tensors
            all_samples.append(transformed_sequence)  # List of [B, T, D] tensors

            # Normalize attention weights across all sequences
            attn_weights_concat = tf.concat(all_attn_weights, axis=1)  # [B, L * T, 1]
            normalize_factor = tf.maximum(tf.reduce_sum(attn_weights_concat, axis=1, keepdims=True), SMALL_NUMBER)
            normalized_attn_weights = attn_weights_concat / normalize_factor  # [B, L * T, 1] 

            # Compute the weighted average
            weighted_sequence = tf.concat(all_samples, axis=1) * normalized_attn_weights  # [B, L * T, D]
            aggregated_sequence = tf.reduce_sum(weighted_sequence, axis=1)  # [B, L * T, D]

            self._ops['pooled_{0}'.format(i)] = aggregated_sequence

            # [B, K]
            output_size = num_output_features if self.output_type != OutputType.MULTI_CLASSIFICATION else self.metadata[NUM_CLASSES]
            level_output, output_hidden = mlp(inputs=aggregated_sequence,
                                  output_size=output_size,
                                  hidden_sizes=self.hypers.model_params.get('output_hidden_units'),
                                  activations=self.hypers.model_params['output_hidden_activation'],
                                  dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                  name=output_layer_name,
                                  compression_seed=OUTPUT_SEED,
                                  compression_fraction=compression_fraction)

            self._ops['hidden0_{0}'.format(i)] = output_hidden[0]
            self._ops['hidden1_{0}'.format(i)] = output_hidden[1]

            # Pooling of output logits to directly combine outputs from each level
            if self.hypers.model_params.get('pool_outputs', False):
                # Compute self-attention pooling weight, [B, 1]
                output_attn_weight, _ = mlp(inputs=aggregated_sequence,
                                            output_size=1,
                                            hidden_sizes=[],
                                            activations='sigmoid',
                                            dropout_keep_rate=1.0,
                                            should_bias_final=True,
                                            should_activate_final=True,
                                            name=OUTPUT_ATTENTION)
                output_attn_weights.append(output_attn_weight)  # List of [B, 1] tensors
                level_outputs.append(level_output)  # List of [B, K] tensors

                # [B, L, 1]
                attn_weight_concat = tf.concat(tf.nest.map_structure(lambda t: tf.expand_dims(t, axis=1), output_attn_weights), axis=1)
                normalized_attn_weights = attn_weight_concat / tf.maximum(tf.reduce_sum(attn_weight_concat, axis=1, keepdims=True), SMALL_NUMBER)

                # [B, L, K]
                level_outputs_concat = tf.concat(tf.nest.map_structure(lambda t: tf.expand_dims(t, axis=1), level_outputs), axis=1)

                # [B, K]
                level_output = tf.reduce_sum(level_outputs_concat * normalized_attn_weights, axis=1)

            # Compute the stop output using a dense layer. This results in a [B, 1] array.
            # The state used to compute the stop output depends on the model type (sample or cascade)
            if is_cascade(self.model_type):
                # In the case of cascade models, the state is just the aggregated state
                stop_output_state = aggregated_sequence
            else:
                # In the case of sample models, we use an aggregation of the first states from each computed sequence
                first_states = tf.concat([tf.expand_dims(transformed[:, 0, :], axis=1) for transformed in all_samples], axis=1)  # [B, L, D]
                first_attn_weights = tf.concat([tf.expand_dims(attn[:, 0, :], axis=1) for attn in all_attn_weights], axis=1)  # [B, L, 1]

                # Pool the first states using a weighted average
                normalize_factor = tf.maximum(tf.reduce_sum(first_attn_weights, axis=1, keepdims=True), SMALL_NUMBER)  # [B, 1, 1]
                normalized_attn_weights = first_attn_weights / normalize_factor  # [B, L, 1]
                stop_output_state = tf.reduce_sum(first_states * normalized_attn_weights, axis=1)  # [B, D]

            stop_output, _ = mlp(inputs=stop_output_state,
                                 output_size=1,
                                 hidden_sizes=self.hypers.model_params['stop_output_hidden_units'],
                                 activations=self.hypers.model_params['stop_output_activation'],
                                 should_bias_final=True,
                                 should_activate_final=False,
                                 dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                 name=STOP_PREDICTION,
                                 compression_fraction=None)
            stop_output = tf.squeeze(stop_output, axis=-1)  # [B]
            self._ops[stop_output_name] = tf.math.sigmoid(stop_output)  # [B]
            stop_outputs.append(stop_output)

            if self.output_type == OutputType.BINARY_CLASSIFICATION:
                classification_output = compute_binary_classification_output(model_output=level_output,
                                                                             labels=self._placeholders[OUTPUT])

                self._ops[logits_name] = classification_output.logits
                self._ops[prediction_name] = classification_output.predictions
                self._ops[accuracy_name] = classification_output.accuracy
                self._ops[f1_score_name] = classification_output.f1_score
            elif self.output_type == OutputType.MULTI_CLASSIFICATION:
                classification_output = compute_multi_classification_output(model_output=level_output,
                                                                            labels=self._placeholders[OUTPUT])
                self._ops[logits_name] = classification_output.logits
                self._ops[prediction_name] = classification_output.predictions
                self._ops[accuracy_name] = classification_output.accuracy
                self._ops[f1_score_name] = classification_output.f1_score
            else:
                self._ops[prediction_name] = level_output

            outputs.append(level_output)

        combined_outputs = tf.concat(tf.nest.map_structure(lambda t: tf.expand_dims(t, axis=1), outputs), axis=1)
        self._ops[ALL_PREDICTIONS_NAME] = combined_outputs

        combined_stop_outputs = tf.concat(tf.nest.map_structure(lambda t: tf.expand_dims(t, axis=1), stop_outputs), axis=1)
        self._ops[STOP_OUTPUT_NAME] = combined_stop_outputs

    def _make_rnn_model(self, is_train: bool):
        """
        Builds an Adaptive RNN model.
        """
        outputs: List[tf.Tensor] = []
        level_outputs: List[tf.Tensor] = []  # Only used when pool_outputs is True
        output_attn_weights: List[tf.Tensor] = []  # Only used when pool_outputs is True
        stop_outputs: List[tf.Tensor] = []
        states_list: List[tf.TensorArray] = []
        bw_states_list: List[tf.TensorArray] = []
        prev_state: Optional[tf.Tensor] = None

        num_output_features = self.metadata[NUM_OUTPUT_FEATURES]

        for i in range(self.num_sequences):
            # Get relevant variable names
            input_name = get_input_name(i)
            cell_name = get_cell_level_name(i, self.hypers.model_params['share_cell_weights'])
            rnn_level_name = get_rnn_level_name(i)
            output_layer_name = get_output_layer_name(i, self.hypers.model_params['share_output_weights'])
            stop_output_name = get_stop_output_name(i)
            logits_name = get_logits_name(i)
            prediction_name = get_prediction_name(i)
            loss_name = get_loss_name(i)
            gate_name = get_gates_name(i)
            state_name = get_states_name(i)
            accuracy_name = get_accuracy_name(i)
            f1_score_name = get_f1_score_name(i)
            embedding_name = get_embedding_name(i, self.hypers.model_params.get('share_embedding_weights', True))

            # Create the embedding layer. Output is a [B, T, D] tensor where T is the seq length of this level.
            input_sequence, _ = dense(inputs=self._placeholders[input_name],
                                      units=self.hypers.model_params['state_size'],
                                      activation=self.hypers.model_params['embedding_activation'],
                                      use_bias=True,
                                      name=embedding_name,
                                      compression_seed=EMBEDDING_SEED,
                                      compression_fraction=self.hypers.model_params.get('compression_fraction'))

            self._ops['embedding_{0}'.format(i)] = input_sequence

            # Create the RNN Cell
            cell = make_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                 input_units=self.hypers.model_params['state_size'],
                                 output_units=self.hypers.model_params['state_size'],
                                 activation=self.hypers.model_params['rnn_activation'],
                                 num_layers=self.hypers.model_params['rnn_layers'],
                                 name=cell_name,
                                 compression_fraction=self.hypers.model_params.get('compression_fraction'),
                                 compression_seed=TRANSFORM_SEED)

            inputs = input_sequence
            initial_state = cell.zero_state(batch_size=tf.shape(inputs)[0], dtype=tf.float32)

            # Set the initial state for chunked model types
            if prev_state is not None:
                if is_cascade(self.model_type):
                    initial_state = prev_state

            # Set previous states for the Sample model type
            prev_states = None
            if is_sample(self.model_type) and i > 0:
                prev_states = states_list[i-1]

            # Run RNN and collect outputs
            rnn_out = dynamic_rnn(cell=cell,
                                  inputs=inputs,
                                  previous_states=prev_states,
                                  initial_state=initial_state,
                                  name=rnn_level_name,
                                  should_share_weights=self.hypers.model_params['share_rnn_weights'],
                                  fusion_mode=self.hypers.model_params.get('fusion_mode'),
                                  compression_fraction=self.hypers.model_params.get('compression_fraction'))
            rnn_outputs = rnn_out.outputs  # [B, T, D]
            rnn_states = rnn_out.states
            rnn_gates = rnn_out.gates

            # Save previous states
            states_list.append(rnn_states)

            # Get the final state
            last_index = tf.shape(inputs)[1] - 1
            final_output = rnn_outputs.read(index=last_index)
            final_state = rnn_states.read(index=last_index)  # [L, B, D] where L is the number of RNN layers
            final_state = tf.concat(tf.unstack(final_state, axis=0), axis=-1)  # [B, D * L]

            # [B, D]
            rnn_output = pool_rnn_outputs(rnn_outputs, final_state, pool_mode=self.hypers.model_params['pool_mode'])

            self._ops['transformed_{0}'.format(i)] = rnn_outputs.stack()
            self._ops['fusion_{0}'.format(i)] = rnn_out.fusion.stack()

            # If the model is bidirectional, then we also run the backward direction
            if is_bidirectional(self.model_type):

                # Create the Bakcward RNN Cell
                bw_cell_name = get_backward_name(cell_name)
                bw_cell = make_rnn_cell(cell_type=self.hypers.model_params['rnn_cell_type'],
                                        input_units=self.hypers.model_params['state_size'],
                                        output_units=self.hypers.model_params['state_size'],
                                        activation=self.hypers.model_params['rnn_activation'],
                                        num_layers=self.hypers.model_params['rnn_layers'],
                                        name=bw_cell_name,
                                        compression_fraction=self.hypers.model_params.get('compression_fraction'),
                                        compression_seed=get_backward_name(TRANSFORM_SEED))

                # Set previous states
                bw_prev_states = None
                if i > 0 and not is_independent(self.model_type):
                    bw_prev_states = bw_states_list[i-1]

                # Run RNN and collect outputs
                initial_state = cell.zero_state(batch_size=tf.shape(inputs)[0], dtype=tf.float32)
                bw_rnn_name = get_backward_name(rnn_level_name)
                bw_rnn_out = dynamic_rnn(cell=bw_cell,
                                         inputs=inputs,
                                         previous_states=bw_prev_states,
                                         initial_state=initial_state,
                                         name=bw_rnn_name,
                                         should_share_weights=self.hypers.model_params['share_rnn_weights'],
                                         fusion_mode=self.hypers.model_params.get('fusion_mode'),
                                         compression_fraction=self.hypers.model_params.get('compression_fraction'),
                                         should_reverse=True)
                bw_rnn_outputs = bw_rnn_out.outputs  # [B, T, D]
                bw_rnn_states = bw_rnn_out.states
                bw_rnn_gates = bw_rnn_out.gates

                # Save previous states
                bw_states_list.append(bw_rnn_states)

                # Get the final state
                last_index = tf.shape(inputs)[1] - 1
                bw_final_output = bw_rnn_outputs.read(index=last_index)
                bw_final_state = bw_rnn_states.read(index=last_index)  # [L, B, D] where L is the number of RNN layers
                bw_final_state = tf.concat(tf.unstack(bw_final_state, axis=0), axis=-1)  # [B, D * L]

                # [B, D]
                bw_rnn_output = pool_rnn_outputs(bw_rnn_outputs, bw_final_state, pool_mode=self.hypers.model_params['pool_mode'])

                # Combine the forward and backward states
                rnn_output = tf.concat([rnn_output, bw_rnn_output], axis=-1)  # [B, 2 * D]

            # Apply dropout to the pooled RNN state
            rnn_output = tf.nn.dropout(rnn_output, keep_prob=self._placeholders[DROPOUT_KEEP_RATE])

            # [B, K]
            output_size = num_output_features if self.output_type != OutputType.MULTI_CLASSIFICATION else self.metadata[NUM_CLASSES]
            level_output, _ = mlp(inputs=rnn_output,
                                  output_size=output_size,
                                  hidden_sizes=self.hypers.model_params.get('output_hidden_units'),
                                  activations=self.hypers.model_params['output_hidden_activation'],
                                  should_bias_final=True,
                                  dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                  name=output_layer_name,
                                  compression_fraction=self.hypers.model_params.get('compression_fraction'),
                                  compression_seed=OUTPUT_SEED)

            if self.hypers.model_params.get('pool_outputs', False):
                # Compute self-attention pooling weight, [B, 1]
                output_attn_weight, _ = mlp(inputs=rnn_output,
                                            output_size=1,
                                            hidden_sizes=[],
                                            activations='sigmoid',
                                            dropout_keep_rate=1.0,
                                            should_bias_final=True,
                                            should_activate_final=True,
                                            name=OUTPUT_ATTENTION)
                output_attn_weights.append(output_attn_weight)  # List of [B, 1] tensors
                level_outputs.append(level_output)  # List of [B, K] tensors

                # [B, L, 1]
                attn_weight_concat = tf.concat(tf.nest.map_structure(lambda t: tf.expand_dims(t, axis=1), output_attn_weights), axis=1)
                normalized_attn_weights = attn_weight_concat / tf.maximum(tf.reduce_sum(attn_weight_concat, axis=1, keepdims=True), SMALL_NUMBER)

                # [B, L, K]
                level_outputs_concat = tf.concat(tf.nest.map_structure(lambda t: tf.expand_dims(t, axis=1), level_outputs), axis=1)

                # [B, K]
                level_output = tf.reduce_sum(level_outputs_concat * normalized_attn_weights, axis=1)

            # Append the output (pooled or not) to the list of outputs
            outputs.append(level_output)

            # Get the state for the stop output (dependent no the model type)
            state_index = tf.shape(inputs)[1] - 1 if is_cascade(self.model_type) else 0 
            stop_output_state = rnn_states.read(index=state_index)
            stop_output_state = tf.concat(tf.unstack(stop_output_state, axis=0), axis=-1)  # [B, D * L]

            # Compute the stop output using a single dense layer. This is a [B, 1] array.
            stop_output, _ = mlp(inputs=stop_output_state,
                                 output_size=1,
                                 hidden_sizes=self.hypers.model_params['stop_output_hidden_units'],
                                 activations=self.hypers.model_params['stop_output_activation'],
                                 should_bias_final=True,
                                 should_activate_final=False,
                                 dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                 name=STOP_PREDICTION,
                                 compression_fraction=None)
            stop_output = tf.squeeze(stop_output, axis=-1)  # [B]
            self._ops[stop_output_name] = tf.math.sigmoid(stop_output)  # [B]
            stop_outputs.append(stop_output)

            if self.output_type == OutputType.BINARY_CLASSIFICATION:
                classification_output = compute_binary_classification_output(model_output=level_output,
                                                                             labels=self._placeholders[OUTPUT])

                self._ops[logits_name] = classification_output.logits
                self._ops[prediction_name] = classification_output.predictions
                self._ops[accuracy_name] = classification_output.accuracy
                self._ops[f1_score_name] = classification_output.f1_score
            elif self.output_type == OutputType.MULTI_CLASSIFICATION:
                classification_output = compute_multi_classification_output(model_output=level_output,
                                                                            labels=self._placeholders[OUTPUT])
                self._ops[logits_name] = classification_output.logits
                self._ops[prediction_name] = classification_output.predictions
                self._ops[accuracy_name] = classification_output.accuracy
                self._ops[f1_score_name] = classification_output.f1_score
            else:
                self._ops[prediction_name] = level_output

            self._ops[gate_name] = rnn_gates.stack()  # [B, T, M, D]
            self._ops[state_name] = rnn_states.stack()  # [B, T, D]

            # Save previous state for possible reuse at the next level
            prev_state = rnn_states.read(index=last_index)

        combined_outputs = tf.concat(tf.nest.map_structure(lambda t: tf.expand_dims(t, axis=1), outputs), axis=1)
        self._ops[ALL_PREDICTIONS_NAME] = combined_outputs

        combined_stop_outputs = tf.concat(tf.nest.map_structure(lambda t: tf.expand_dims(t, axis=1), stop_outputs), axis=1)
        self._ops[STOP_OUTPUT_NAME] = combined_stop_outputs


    def make_loss(self):
        losses: List[tf.Tensor] = []

        loss_mode = self.hypers.model_params['loss_mode'].lower()

        # The loss_op keys are ordered by the output level
        for level in range(self.num_outputs):
            loss_name = get_loss_name(level)
            expected_output = self._placeholders[OUTPUT]
            predictions = self._ops[get_prediction_name(level)]

            if self.output_type == OutputType.BINARY_CLASSIFICATION:
                logits = self.ops[get_logits_name(level)]
                predicted_probs = tf.math.sigmoid(logits)

                if loss_mode in ('cross-entropy', 'cross-entropy', 'accuracy'):
                    sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=expected_output,
                                                                          logits=logits)
                    self._ops[loss_name] = tf.reduce_mean(sample_loss)
                elif loss_mode in ('f1', 'f1_score', 'f1-score'):
                    self._ops[loss_name] = f1_score_loss(predicted_probs=predicted_probs, labels=expected_output)
                else:
                    raise ValueError(f'Unknown loss mode: {loss_mode}')
            elif self.output_type == OutputType.MULTI_CLASSIFICATION:
                logits = self.ops[get_logits_name(level)]
                labels = tf.squeeze(expected_output, axis=-1)

                sample_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
                self._ops[loss_name] = tf.reduce_mean(sample_loss)
            else:  # Regression task
                self._ops[loss_name] = tf.reduce_mean(tf.square(predictions - expected_output))  # MSE loss

            batch_loss = self._ops[loss_name]
            losses.append(tf.reduce_mean(batch_loss))

        losses = tf.stack(losses)  # [N], N is the number of sequences
        weighted_losses = tf.reduce_sum(losses * self._placeholders['loss_weights'], axis=-1)  # Scalar
        self._ops[LOSS] = weighted_losses

        # Add loss from stop layer
        if self.hypers.model_params.get('use_stop_layer', False):
            stop_outputs = self._ops[STOP_OUTPUT_NAME]  # [B, L]

            # [B, L]
            all_predictions = tf.cast(tf.argmax(self._ops[ALL_PREDICTIONS_NAME], axis=-1), dtype=tf.int32)  # [B, L]
            stop_labels = tf.cast(tf.equal(all_predictions, self._placeholders[OUTPUT]), dtype=tf.float32)  # [B, L]

            # Compute binary cross entropy loss and sum over levels, average over batch
            stop_element_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=stop_outputs, labels=stop_labels)

            stop_loss = self._placeholders[STOP_LOSS_WEIGHT] * tf.reduce_mean(tf.reduce_sum(stop_element_loss, axis=-1))
            self._ops[LOSS] += stop_loss

        # Add any regularization to the loss function
        reg_loss = self.regularize_weights(name=self.hypers.model_params.get('regularization_name'),
                                           scale=self.hypers.model_params.get('regularization_scale', 0.0))
        if reg_loss is not None:
            self._ops[LOSS] += reg_loss

    def anytime_generator(self, feed_dict: Dict[tf.Tensor, List[Any]],
                          max_num_levels: int) -> Optional[Iterable[np.ndarray]]:
        """
        Anytime Inference in a generator-like fashion
        """
        with self.sess.graph.as_default():
            num_levels = max(self.num_outputs, max_num_levels) if max_num_levels is not None else self.num_outputs

            # Setup the partial run with the output operations and input placeholders
            prediction_ops = [self.ops[get_prediction_name(i)] for i in range(num_levels)]

            if self.output_type == OutputType.REGRESSION:
                logit_ops = []
            else:
                logit_ops = [self.ops[get_logits_name(i)] for i in range(num_levels)]

            input_names = set((get_input_name(i) for i in range(num_levels)))
            placeholders = [ph for name, ph in self.placeholders.items() if name in input_names or name == DROPOUT_KEEP_RATE]

            ops = prediction_ops + logit_ops
            handle = self.sess.partial_run_setup(ops, placeholders)

            for level in range(num_levels):
                prediction_op = self.ops[get_prediction_name(level)]
                logit_op = self.ops.get(get_logits_name(level))

                # Form input dictionary with this level's input sequence
                input_dict = {ph: val for ph, val in feed_dict.items() if ph.name.startswith(get_input_name(level))}

                # Specify dropout at level 0 to avoid feeding multiple times. We explicitly turn off dropout
                # at inference time.
                if level == 0:
                    input_dict[self.placeholders[DROPOUT_KEEP_RATE]] = 1.0

                logits = None
                if self.output_type == OutputType.REGRESSION:
                    predictions = self.sess.partial_run(handle, fetches=[prediction_op], feed_dict=input_dict)
                else:
                    predictions, logits = self.sess.partial_run(handle, fetches=[prediction_op, logit_op], feed_dict=input_dict)

                yield predictions, logits
