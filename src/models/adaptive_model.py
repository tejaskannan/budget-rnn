import tensorflow as tf
import numpy as np
from collections import defaultdict
from typing import List, Optional, Tuple, Dict, Any, DefaultDict, Iterable

from models.tf_model import TFModel
from layers.dense import mlp, dense
from layers.cells.cell_factory import make_rnn_cell, CellClass, CellType
from layers.output_layers import OutputType, compute_binary_classification_output, compute_multi_classification_output
from dataset.dataset import Dataset, DataSeries
from utils.hyperparameters import HyperParameters
from utils.tfutils import mask_last_element, successive_pooling
from utils.constants import SMALL_NUMBER, BIG_NUMBER, ACCURACY, OUTPUT, INPUTS, LOSS, OPTIMIZER_OP
from utils.constants import DROPOUT_KEEP_RATE, MODEL, NUM_CLASSES, GLOBAL_STEP, PREDICTION, LOGITS
from utils.constants import INPUT_SHAPE, NUM_OUTPUT_FEATURES, SEQ_LENGTH, INPUT_NOISE, STOP_LOSS_WEIGHT
from utils.constants import EMBEDDING_NAME, TRANSFORM_NAME, AGGREGATION_NAME, OUTPUT_LAYER_NAME, LOSS_WEIGHTS
from utils.constants import STOP_OUTPUT_NAME, STOP_OUTPUT_LOGITS, STOP_PREDICTION, RNN_CELL_NAME
from utils.loss_utils import get_loss_weights, get_temperate_loss_weight
from utils.sequence_model_utils import SequenceModelType, is_rnn, is_nbow, is_sample
from utils.testing_utils import ClassificationMetric, RegressionMetric, get_binary_classification_metric, get_regression_metric, get_multi_classification_metric


class AdaptiveModel(TFModel):

    def __init__(self, hyper_parameters: HyperParameters, save_folder: str, is_train: bool):
        super().__init__(hyper_parameters, save_folder, is_train)

        model_type = self.hypers.model_params['model_type'].upper()
        self.model_type = SequenceModelType[model_type]

        self.name = model_type

    @property
    def stride_length(self) -> int:
        return int(self.hypers.model_params['stride_length'])

    @property
    def seq_length(self) -> int:
        return self.metadata[SEQ_LENGTH]

    @property
    def samples_per_seq(self) -> int:
        return int(self.seq_length / self.stride_length)

    @property
    def num_outputs(self) -> int:
        return self.seq_length

    @property
    def num_stop_outputs(self) -> int:
        return 1 if self.stride_length == 1 else 2

    @property
    def num_output_features(self) -> int:
        if self.output_type == OutputType.MULTI_CLASSIFICATION:
            return self.metadata[NUM_CLASSES]
        return self.metadata[NUM_OUTPUT_FEATURES]

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
        end_stop_loss_weight = self.hypers.model_params.get(STOP_LOSS_WEIGHT, 0.0)
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

        predictions_array = np.vstack(predictions).astype(int)  # [N, T]
        labels_array = np.vstack(labels).reshape(-1).astype(int)  # [N]

        result: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)
        for seq_idx in range(self.seq_length):
            for metric_name in ClassificationMetric:
                if self.output_type == OutputType.BINARY_CLASSIFICATION:
                    metric_value = get_binary_classification_metric(metric_name, predictions_array[:, seq_idx], labels_array)
                else:
                    metric_value = get_multi_classification_metric(metric_name, predictions_array[:, seq_idx], labels_array, self.metadata[NUM_CLASSES])

                result['{0}_{1}'.format(PREDICTION, seq_idx)][metric_name.name] = metric_value

        return result

    def make_model(self, is_train: bool):
        with tf.variable_scope(MODEL, reuse=tf.AUTO_REUSE):
            if is_nbow(self.model_type):
                self._make_nbow_model(is_train)
            else:
                self._make_rnn_model(is_train)

    def _make_nbow_model(self, is_train: bool):
        state_size = self.hypers.model_params['state_size']

        # Apply the embedding layer, [B, T, D]
        embedding, _ = dense(inputs=self._placeholders[INPUTS],
                             units=state_size,
                             activation=self.hypers.model_params['embedding_activation'],
                             use_bias=True,
                             name=EMBEDDING_NAME)

        # Apply the transformation layer, [B, T, D]
        transformed, _ = mlp(inputs=embedding,
                             output_size=state_size,
                             hidden_sizes=self.hypers.model_params['transform_units'],
                             activations=self.hypers.model_params['transform_activation'],
                             dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                             should_activate_final=True,
                             should_bias_final=True,
                             should_dropout_final=True,
                             name=TRANSFORM_NAME)

        # Compute the attention aggregation weights, [B, T, 1]
        aggregation_weights, _ = dense(inputs=transformed,
                                       output_size=1,
                                       activation='sigmoid',
                                       use_bias=True,
                                       name=AGGREGATION_NAME)

        # For stride lengths > 1, we re-order the tensor based on the subsequences
        if self.stride_length > 1:
            subsequence_indices = np.arange(start=0, stop=self.seq_length, step=self.stride_length)
            subsequence_indices = np.tile(subsequence_indices, reps=(self.stride_length, ))  # [T]

            offsets = np.repeat(np.arange(start=0, stop=self.stride_length), repeats=self.samples_per_seq)  # [T]

            sequence_indices = subsequence_indices + offsets  # [T]

            # Apply the sub-sequence shuffling
            transformed = tf.gather(transformed, indices=sequence_indices, axis=1)  # [B, T, D]
            aggregation_weights = tf.gather(aggregation_weights, indices=sequence_indices, axis=1)  # [B, T, 1]

        # Pool the transformed states, [B, T, D] output
        pooled_states = successive_pooling(transformed, aggregation_weights, self.seq_length, name=AGGREGATION_NAME)

        # For stride lengths > 1, we undo the shuffling for consistency purposes
        if self.stride_length > 1:
            pooled_states = tf.gather(pooled_states, indices=sequence_indices, axis=1)  # [B, T, D]

        # Create the output prediction, [B, K]
        output, _ = mlp(inputs=pooled_states,
                        output_size=self.num_output_features,
                        hidden_sizes=self.hypers.model_params['output_hidden_units'],
                        activations=self.hypers.model_params['output_hidden_activation'],
                        dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                        should_activate_final=False,
                        should_bias_final=True,
                        should_dropout_final=False,
                        name=OUTPUT_LAYER_NAME)

        # Create the stop output, [B, T, 1] or [B, T, 2] based on the stride length
        stop_output_logits, _ = mlp(inputs=pooled_states,
                                    output_size=self.num_stop_outputs,
                                    hidden_sizes=self.hypers.model_params['stop_output_units'],
                                    activations=self.hypers.model_params['stop_output_activation'],
                                    dropout_keep_rate=self._placeholders[DROPOUT_KEEP_RATE],
                                    should_activate_final=False,
                                    should_bias_final=True,
                                    should_dropout_final=False,
                                    name=STOP_PREDICTION)
        self.ops[STOP_OUTPUT_LOGITS] = stop_output_logits
        self.ops[STOP_OUTPUT_NAME] = tf.math.sigmoid(stop_output_logits)

        # Expand dimensions on the expected output for later broadcasting
        expected_output = tf.expand_dims(self._placeholders[OUTPUT], axis=1)  # [B, 1, 1]

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

    def _make_rnn_model(self, is_train: bool):
        """
        Builds an Adaptive RNN Model.
        """
        state_size = self.hypers.model_params['state_size']
        batch_size = tf.shape(self._placeholders[INPUTS])[0]

        # Compute the input embeddings, result is a [B, T, D] tensor
        embeddings, _ = dense(inputs=self._placeholders[INPUTS],
                              units=state_size,
                              activation=self.hypers.model_params['embedding_activation'],
                              use_bias=True,
                              name=EMBEDDING_NAME)

        # Create the RNN Cell
        rnn_cell_class = CellClass.STANDARD if self.stride_length == 1 else CellClass.SAMPLE
        rnn_cell = make_rnn_cell(cell_class=rnn_cell_class,
                                 cell_type=CellType[self.hypers.model_params['rnn_cell_type'].upper()],
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
            prev_states = tf.get_variable(name='prev-states',
                                          initializer=tf.zeros_initializer(),
                                          shape=[1, 1, state_size],
                                          dtype=tf.float32,
                                          trainable=False)
            prev_states = tf.tile(prev_states, multiples=(batch_size, self.samples_per_seq, 1))  # [B, L, D]

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
            offsets = np.repeat(offsets, repeats=self.samples_per_seq)  # [T]

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
                        output_size=self.num_output_features,
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

            batch_idx = tf.expand_dims(tf.range(start=0, limit=batch_size) * self.stride_length, axis=1)  # [B, 1]
            subseq_idx = tf.tile(tf.range(start=0, limit=self.stride_length, dtype=tf.int32),
                                 multiples=[self.samples_per_seq])  # [T]
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

        # Create the loss operation
        self._ops[LOSS] = weighted_loss + self._placeholders[STOP_LOSS_WEIGHT] * (stop_loss + stop_level_loss)
