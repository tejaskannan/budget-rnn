import tensorflow as tf
import numpy as np
import re
import time
from collections import namedtuple, defaultdict, OrderedDict
from typing import List, Optional, Tuple, Dict, Any, Set, Union, DefaultDict, Iterable

from models.tf_model import TFModel
from layers.basic import rnn_cell, mlp, dense
from layers.cells.cell_factory import make_rnn_cell
from layers.output_layers import OutputType, compute_binary_classification_output, compute_multi_classification_output
from dataset.dataset import Dataset, DataSeries
from utils.hyperparameters import HyperParameters
from utils.tfutils import mask_last_element
from utils.constants import SMALL_NUMBER, BIG_NUMBER, ACCURACY, OUTPUT, INPUTS, LOSS, OUTPUT_SEED, OPTIMIZER_OP, GLOBAL_STEP
from utils.constants import NODE_REGEX_FORMAT, DROPOUT_KEEP_RATE, MODEL, SCHEDULED_MODEL, NUM_CLASSES, TRANSFORM_SEED
from utils.constants import INPUT_SHAPE, NUM_OUTPUT_FEATURES, SEQ_LENGTH, INPUT_NOISE, EMBEDDING_SEED, AGGREGATE_SEED, STOP_LOSS_WEIGHT
from utils.loss_utils import get_loss_weights, get_temperate_loss_weight
from utils.rnn_utils import *
from utils.testing_utils import ClassificationMetric, RegressionMetric, get_binary_classification_metric, get_regression_metric, get_multi_classification_metric


LOSS_WEIGHTS = 'loss_weights'


class AdaptiveModel(TFModel):

    def __init__(self, hyper_parameters: HyperParameters, save_folder: str, is_train: bool):
        super().__init__(hyper_parameters, save_folder, is_train)

        model_type = self.hypers.model_params['model_type'].upper()
        self.model_type = SequenceModelType[model_type]

        self.name = model_type

    @property
    def stride_length(self) -> float:
        return self.hypers.model_params['stride_length']

    @property
    def seq_length(self) -> int:
        return self.metadata[SEQ_LENGTH]

    @property
    def num_outputs(self) -> int:
        return self.seq_length

    @property
    def num_stop_outputs(self) -> int:
        return 1 if self.stride_length == 1 else 2

    @property
    def accuracy_op_names(self) -> List[str]:
        return [ACCURACY]

    @property
    def prediction_ops(self) -> List[str]:
        return [PREDICTION]

    @property
    def logit_op_names(self) -> List[str]:
        return [LOGITS]

    @property
    def output_ops(self) -> List[str]:
        return self.prediction_ops

    @property
    def loss_op_names(self) -> List[str]:
        return [LOSS]

    @property
    def optimizer_op_names(self) -> List[str]:
        return [OPTIMIZER_OP]

    @property
    def global_step_op_names(self) -> List[str]:
        return [GLOBAL_STEP]

    def batch_to_feed_dict(self, batch: Dict[str, List[Any]], is_train: bool, epoch_num: int) -> Dict[tf.Tensor, np.ndarray]:
        dropout = self.hypers.dropout_keep_rate if is_train else 1.0
        input_batch = np.array(batch[INPUTS])
        output_batch = np.array(batch[OUTPUT])

        if input_batch.shape[1] == 1:
            input_batch = np.squeeze(input_batch, axis=1)

        input_shape = self.metadata[INPUT_SHAPE]
        num_output_features = self.metadata[NUM_OUTPUT_FEATURES]

        # Calculate the stop loss weight based on on the epoch number. The weight is increased
        # exponentially per epoch and reaches the final value after Patience steps.
        end_stop_loss_weight = self.hypers.model_params.get('stop_loss_weight', 0.0)
        stop_loss_weight = get_temperate_loss_weight(start_weight=1e-5,
                                                     end_weight=end_stop_loss_weight,
                                                     step=epoch_num,
                                                     max_steps=self.hypers.patience - 1)

        feed_dict = {
            self._placeholders[INPUTS]: input_batch,
            self._placeholders[OUTPUT]: output_batch.reshape(-1, num_output_features),
            self._placeholders[DROPOUT_KEEP_RATE]: dropout,
            self._placeholders[STOP_LOSS_WEIGHT]: stop_loss_weight
        }

        # The loss weights are sorted in ascending order to more-heavily weight larger sample sizes.
        # This operation is included to prevent bugs as we intuitively want to increase
        # accuracy with larger samples.
        loss_weights = get_loss_weights(n=self.num_outputs, mode=self.hypers.model_params.get(LOSS_WEIGHTS))
        feed_dict[self._placeholders[LOSS_WEIGHTS]] = loss_weights  # Normalize the loss weights

        return feed_dict

    def make_placeholders(self, is_frozen: bool = False):
        """
        Create model placeholders.
        """
        # Extract parameters
        input_features_shape = self.metadata[INPUT_SHAPE]
        num_output_features = self.metadata[NUM_OUTPUT_FEATURES]

        output_dtype = tf.int32 if self.output_type == OutputType.MULTI_CLASSIFICATION else tf.float32

        if not is_frozen:
            self._placeholders[INPUTS] = tf.placeholder(shape=[None, self.seq_length] + list(input_features_shape),
                                                        dtype=tf.float32,
                                                        name=INPUTS)
            # [B, K]
            self._placeholders[OUTPUT] = tf.placeholder(shape=[None, num_output_features],
                                                        dtype=output_dtype,
                                                        name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = tf.placeholder(shape=[],
                                                                   dtype=tf.float32,
                                                                   name=DROPOUT_KEEP_RATE)
            self._placeholders[LOSS_WEIGHTS] = tf.placeholder(shape=[self.num_outputs],
                                                                dtype=tf.float32,
                                                                name=LOSS_WEIGHTS)
            self._placeholders[STOP_LOSS_WEIGHT] = tf.placeholder(shape=[],
                                                                  dtype=tf.float32,
                                                                  name=STOP_LOSS_WEIGHT)
        else:
            self._placeholders[INPUTS] = tf.ones(shape=[1, self.seq_length] + list(input_features_shape), dtype=tf.float32, name=INPUTS)
            self._placeholders[OUTPUT] = tf.ones(shape=[1, num_output_features], dtype=output_dtype, name=OUTPUT)
            self._placeholders[DROPOUT_KEEP_RATE] = tf.ones(shape=[], dtype=tf.float32, name=DROPOUT_KEEP_RATE)
            self._placeholders[STOP_LOSS_WEIGHT] = tf.ones(shape=[], dtype=tf.float32, name=STOP_LOSS_WEIGHT)
            self._placeholders[LOSS_WEIGHTS] = tf.ones(shape=[self.num_outputs], dtype=tf.float32, name=LOSS_WEIGHTS)

    def predict_classification(self, test_batch_generator: Iterable[Any],
                               batch_size: int,
                               max_num_batches: Optional[int]) -> DefaultDict[str, Dict[str, Any]]:
        predictions: List[np.ndarray] = []
        labels: List[np.ndarray] = []

        for batch_num, batch in enumerate(test_batch_generator):
            if max_num_batches is not None and batch_num >= max_num_batches:
                break

            feed_dict = self.batch_to_feed_dict(batch, is_train=False, epoch_num=0)
            results = self.execute(ops=self.prediction_ops, feed_dict=feed_dict)

            predictions.append(results[PREDICTION])
            labels.append(np.vstack(batch[OUTPUT]))

        predictions = np.vstack(predictions).astype(int)  # [N, T]
        labels = np.vstack(labels).reshape(-1).astype(int)  # [N]

        result = defaultdict(dict)
        for seq_idx in range(self.seq_length):
            for metric_name in ClassificationMetric:
                if self.output_type == OutputType.BINARY_CLASSIFICATION:
                    metric_value = get_binary_classification_metric(metric_name, predictions[:, seq_idx], labels)
                else:
                    metric_value = get_multi_classification_metric(metric_name, predictions[:, seq_idx], labels, self.metadata[NUM_CLASSES])

                result['{0}_{1}'.format(PREDICTION, seq_idx)][metric_name.name] = metric_value

        return result

    def make_model(self, is_train: bool):
        with tf.variable_scope(MODEL, reuse=tf.AUTO_REUSE):
            if is_nbow(self.model_type):
                self._make_nbow_model(is_train)
            else:
                self._make_rnn_model(is_train)

    def _make_nbow_model(self, is_train: bool):
        outputs: List[tf.Tensor] = []
        stop_outputs: List[tf.Tensor] = []
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

            # Compute attention weights for aggregation. We only compute the
            # weights for this sequence to avoid redundant computation. [B, T, 1] tensor.
            attn_weights, _ = dense(inputs=transformed_sequence,
                                    units=1,
                                    activation=self.hypers.model_params['attn_activation'],
                                    use_bias=True,
                                    name=aggregation_name,
                                    compression_seed=AGGREGATE_SEED,
                                    compression_fraction=compression_fraction)

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

            # [B, K]
            output_size = num_output_features if self.output_type != OutputType.MULTI_CLASSIFICATION else self.metadata[NUM_CLASSES]
            level_output, _ = mlp(inputs=aggregated_sequence,
                                  output_size=output_size,
                                  hidden_sizes=self.hypers.model_params.get('output_hidden_units'),
                                  activations=self.hypers.model_params['output_hidden_activation'],
                                  dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                  should_bias_final=True,
                                  name=output_layer_name,
                                  compression_seed=OUTPUT_SEED,
                                  compression_fraction=compression_fraction)

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
        Builds an Adaptive Cascade RNN Model.
        """
        # Unpack various settings
        if self.output_type == OutputType.MULTI_CLASSIFICATION:
            num_output_features = self.metadata[NUM_CLASSES]
        else:
            num_output_features = self.metadata[NUM_OUTPUT_FEATURES]

        state_size = self.hypers.model_params['state_size']
        batch_size = tf.shape(self._placeholders[INPUTS])[0]

        # Compute the input embeddings, result is a [B, T, D] tensor
        embeddings, _ = dense(inputs=self._placeholders[INPUTS],
                              units=state_size,
                              activation=self.hypers.model_params['embedding_activation'],
                              use_bias=True,
                              name=EMBEDDING_NAME)

        # Create the RNN Cell
        rnn_cell_class = 'standard' if self.stride_length == 1 else 'sample'
        rnn_cell = make_rnn_cell(cell_class=rnn_cell_class,
                                 cell_type=self.hypers.model_params['rnn_cell_type'],
                                 units=state_size,
                                 activation=self.hypers.model_params['rnn_activation'],
                                 name=RNN_CELL_NAME)

        # Execute the RNN, outputs consist of a [B, T, D] tensor
        if self.stride_length == 1:
            initial_state = rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            rnn_outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell,
                                               inputs=embeddings,
                                               initial_state=initial_state,
                                               dtype=tf.float32,
                                               scope=TRANSFORM_NAME)
            transformed = rnn_outputs  # [B, T, D]
            stop_states = rnn_outputs[:, 0:-1, :]  # [B, T - 1, D]
        else:
            samples_per_seq = int(self.seq_length / self.stride_length)

            prev_states = tf.get_variable(name='prev-states',
                                          initializer=tf.zeros_initializer(),
                                          shape=[1, 1, state_size],
                                          dtype=tf.float32,
                                          trainable=False)
            prev_states = tf.tile(prev_states, multiples=(batch_size, samples_per_seq, 1))  # [B, L, D]

            level_outputs: List[tf.Tensor] = []
            for i in range(self.stride_length):
                # Get the inputs for the current sub-sequence
                level_indices = list(range(i, self.seq_length, self.stride_length))
                level_embeddings = tf.gather(embeddings, indices=level_indices, axis=1)  # [B, L, D]

                # Construct the RNN inputs by concatenating the inputs with the previous states, [B, L, 2*D]
                rnn_inputs = tf.concat([level_embeddings, prev_states], axis=-1)

                # Apply the RNN to each sub-sequence, result is a [B, L, D] tensor
                initial_state = rnn_cell.get_initial_state(inputs=level_embeddings, batch_size=batch_size, dtype=tf.float32)
                rnn_outputs, _ = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                   inputs=rnn_inputs,
                                                   initial_state=initial_state,
                                                   dtype=tf.float32,
                                                   scope=TRANSFORM_NAME)
                level_outputs.append(rnn_outputs)

                # Set sequence of previous states
                prev_states = rnn_outputs

            # Concatenate the outputs from each sub-sequence into a [B, T, D] tensor. At this point
            # the sequence elements are ordered with respect to the sub-sequences as opposed to the original
            # sequence. We re-order this tensor back to the original sequence ordering for consistency purposes.
            concat_transformed = tf.concat(level_outputs, axis=1)

            subseq_indices = np.arange(0, stop=self.seq_length, step=self.stride_length)  # [L]
            subseq_indices = np.tile(subseq_indices, reps=self.stride_length)  # [T]

            offsets = np.arange(0, stop=self.stride_length)
            offsets = np.repeat(offsets, repeats=samples_per_seq)  # [T]

            original_indices = subseq_indices + offsets  # [T]

            # [B, T, D]
            transformed = tf.gather(concat_transformed, indices=original_indices, axis=1)

        # Compute the stop output, Result is a [B, T, 1] or [B, T, 2] tensor.
        stop_output, _ = mlp(inputs=transformed,
                             output_size=self.num_stop_outputs,
                             hidden_sizes=self.hypers.model_params['stop_output_hidden_units'],
                             activations=self.hypers.model_params['stop_output_activation'],
                             should_bias_final=True,
                             should_activate_final=False,
                             dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                             name=STOP_PREDICTION)
        self._ops[STOP_OUTPUT_LOGITS] = stop_output
        self._ops[STOP_OUTPUT_NAME] = tf.math.sigmoid(stop_output)  # [B, T, 1] or [B, T, 2]

        # Compute the predictions, Result is a [B, T, K] tensor
        output, _ = mlp(inputs=transformed,
                        output_size=num_output_features,
                        hidden_sizes=self.hypers.model_params['output_hidden_units'],
                        activations=self.hypers.model_params['output_hidden_activation'],
                        should_bias_final=True,
                        should_activate_final=False,
                        dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                        name=OUTPUT_LAYER_NAME)

        # Reshape to [B, 1, 1]
        expected_output = tf.expand_dims(self._placeholders[OUTPUT], axis=-1)

        # Compute the output values
        if self.output_type == OutputType.BINARY_CLASSIFICATION:
            classification_output = compute_binary_classification_output(model_output=output,
                                                                         labels=expected_output)
            self._ops[LOGITS] = classification_output.logits
            self._ops[PREDICTION] = classification_output.predictions
            self._ops[ACCURACY] = classification_output.accuracy
        elif self.output_type == OutputType.MULTI_CLASSIFICATION:
            classification_output = compute_multi_classification_output(model_output=output,
                                                                        labels=expected_output)
            self._ops[LOGITS] = classification_output.logits
            self._ops[PREDICTION] = classification_output.predictions
            self._ops[ACCURACY] = classification_output.accuracy
        else:
            self._ops[PREDICTION] = output

    def make_loss(self):
        """
        Constructs the loss function for this model
        """
        losses: List[tf.Tensor] = []
        expected_output = self._placeholders[OUTPUT]  # [B, 1]

        if self.output_type == OutputType.BINARY_CLASSIFICATION:
            expected_output = tf.tile(expected_output, multiples=(1, self.num_outputs))  # [B, T]
            sample_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=expected_output,
                                                                  logits=self._ops[LOGITS])
        elif self.output_type == OutputType.MULTI_CLASSIFICATION:
            expected_output = tf.tile(expected_output, multiples=(1, self.num_outputs))  # [B, T]
            sample_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=expected_output,
                                                                         logits=self._ops[LOGITS])
        else:
            sample_loss = tf.reduce_sum(tf.square(self._ops[PREDICTION] - expected_output), axis=-1)

        output_loss = tf.reduce_mean(sample_loss, axis=0)  # [T]
        weighted_loss = tf.reduce_sum(output_loss * self._placeholders[LOSS_WEIGHTS])  # Scalar

        predictions = self._ops[PREDICTION]  # [B, T]
        stop_outputs = self._ops[STOP_OUTPUT_LOGITS]  # [B, T, 1] or [B, T, 2]
        stop_labels = tf.cast(tf.equal(predictions, self._placeholders[OUTPUT]), dtype=tf.float32)  # [B, T]

        # For sample models, we have an additional stop output which calculates whether the model is right
        # at each level. We need this signal to prevent capturing intermediate samples which will later
        # be ignored.
        if self.stride_length > 1:
            # Fetch the stop level predictions
            stop_level_pred = stop_outputs[:, :, 1]  # [B, T]

            # Get the output from the first sample in each sub-sequence. We must
            # use the first sample to determine the sub-sequence stopping behavior.
            first_indices = np.arange(0, stop=self.stride_length)  # [L]
            subseq_pred = tf.gather(stop_level_pred, indices=first_indices, axis=1)  # [B, L]

            # Create indices used to for the later segment operation. The segment operation
            # collects the label at each sub-sequence.
            batch_size = tf.shape(stop_level_pred)[0]
            samples_per_seq = int(self.seq_length / self.stride_length)

            batch_idx = tf.expand_dims(tf.range(start=0, limit=batch_size) * self.stride_length, axis=1)  # [B, 1]
            subseq_idx = tf.tile(tf.range(start=0, limit=self.stride_length, dtype=tf.int32),
                                 multiples=[samples_per_seq])  # [T]
            subseq_idx = tf.expand_dims(subseq_idx, axis=0)  # [1, T]
            segment_ids = batch_idx + subseq_idx  # [B, T]

            self._ops['segment_ids'] = segment_ids

            # The stop level labels indicate whether the model is right (1) or wrong (0) at each level.
            # We collect these labels by taking the max over the labels from each sample in each level.
            # This operations results in a [B, L] tensor containing the label for each level (L).
            stop_level_labels = tf.unsorted_segment_max(tf.reshape(stop_labels, shape=[-1]),
                                                        segment_ids=tf.reshape(segment_ids, shape=[-1]),
                                                        num_segments=self.stride_length * batch_size)  # [B, L]
            stop_level_labels = tf.reshape(stop_level_labels, shape=(-1, self.stride_length))  # [B, L]

            # Avoid back-propagating through the stop level labels (this is a not a differentiable operation)
            stop_level_labels = tf.stop_gradient(stop_level_labels)

            self._ops['subseq_pred'] = subseq_pred
            self._ops['stop_level_labels'] = stop_level_labels

            # Compute the cross entropy loss. We mask out the final element because there is no decision
            # to make at the top level. Once we reach the final sub-sequence, we collect all data.
            stop_level_element_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=subseq_pred,
                                                                              labels=stop_level_labels)  # [B, L]
            masked_level_element_loss = mask_last_element(stop_level_element_loss)  # [B, L]

            self._ops['stop_level_loss'] = masked_level_element_loss

            stop_level_loss = tf.reduce_mean(tf.reduce_sum(masked_level_element_loss, axis=-1))  # Scalar
            stop_outputs = stop_outputs[:, :, 0]  # [B, T]
        else:
            stop_level_loss = 0
            stop_outputs = tf.squeeze(stop_outputs, axis=-1)  # [B, T]

        # We explicitly prevent propagating the gradient through the stop labels. These labels are treated
        # as constants with respect to the stop output. This treatment is necessary because the stop labels
        # are not differentiable with respect to the sequence model output.
        stop_labels = tf.stop_gradient(stop_labels)

        self._ops['stop_labels'] = stop_labels

        # Compute binary cross entropy loss and sum over levels, average over batch. We mask out the final output
        # because there is no decision to make at the last sample.
        stop_element_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=stop_outputs, labels=stop_labels)  # [B, T]
        masked_stop_element_loss = mask_last_element(stop_element_loss)  # [B, T]
        stop_loss = tf.reduce_mean(tf.reduce_sum(masked_stop_element_loss, axis=-1))  # Scalar

        self._ops['stop_loss'] = masked_stop_element_loss

        # Create the loss operation
        self._ops[LOSS] = weighted_loss + self._placeholders[STOP_LOSS_WEIGHT] * (stop_loss + stop_level_loss)
