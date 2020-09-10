import tensorflow as tf
import numpy as np
import re
import os
import gc
from datetime import datetime
from collections import defaultdict
from typing import Optional, Iterable, Dict, Any, Union, List, DefaultDict, Set
from sklearn.preprocessing import StandardScaler

from models.base_model import Model
from dataset.dataset import Dataset, DataSeries
from layers.output_layers import OutputType, is_classification
from utils.hyperparameters import HyperParameters
from utils.tfutils import get_optimizer, variables_for_loss_op
from utils.file_utils import read_by_file_suffix, save_by_file_suffix, make_dir
from utils.constants import BIG_NUMBER, NAME_FMT, HYPERS_PATH, GLOBAL_STEP
from utils.constants import METADATA_PATH, MODEL_PATH, TRAIN_LOG_PATH
from utils.constants import LOSS, ACCURACY, OPTIMIZER_OP, INPUTS, OUTPUT, SAMPLE_ID
from utils.constants import TRAIN, VALID, LABEL_MAP, NUM_CLASSES, REV_LABEL_MAP
from utils.constants import INPUT_SHAPE, NUM_OUTPUT_FEATURES, INPUT_SCALER, OUTPUT_SCALER
from utils.constants import SEQ_LENGTH, DROPOUT_KEEP_RATE, MODEL, INPUT_NOISE, SMALL_NUMBER


class TFModel(Model):

    def __init__(self, hyper_parameters: HyperParameters, save_folder: str, is_train: bool):
        super().__init__(hyper_parameters, save_folder, is_train)

        self._sess = tf.Session(graph=tf.Graph())

        self._ops: Dict[str, tf.Tensor] = dict()
        self._placeholders: Dict[str, tf.Tensor] = dict()
        self._is_made = False

        # Get the model output type
        self._output_type = OutputType[self.hypers.model_params['output_type'].upper()]

        # Dictionary with inference operations
        self._inference_ops: Dict[str, tf.Tensor] = dict()

    @property
    def ops(self) -> Dict[str, tf.Tensor]:
        return self._ops

    @property
    def placeholders(self) -> Dict[str, tf.Tensor]:
        return self._placeholders

    @property
    def optimizer_op_name(self) -> str:
        return OPTIMIZER_OP

    @property
    def loss_op_name(self) -> str:
        return LOSS

    @property
    def accuracy_op_name(self) -> str:
        return ACCURACY

    @property
    def output_op_name(self) -> str:
        raise NotImplementedError()

    @property
    def sess(self) -> tf.Session:
        return self._sess

    @property
    def is_made(self) -> bool:
        return self._is_made

    @property
    def trainable_vars(self) -> List[tf.Variable]:
        return list(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    @property
    def output_type(self) -> OutputType:
        return self._output_type

    def load_metadata(self, dataset: Dataset):
        """
        Loads metadata from the dataset. Results are stored
        directly into self.metadata.
        """
        input_samples: List[List[float]] = []
        output_samples: List[List[float]] = []

        # Fetch training samples to prepare for normalization
        unique_labels: Set[Any] = set()
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

            if self.output_type == OutputType.MULTI_CLASSIFICATION:
                unique_labels.add(sample[OUTPUT])

        # Infer the number of input and output features
        first_sample = np.array(input_samples[0])
        input_shape = first_sample.shape[1:]  # Skip the sequence length
        seq_length = len(input_samples[0]) if self.hypers.seq_length is None else self.hypers.seq_length

        # Normalize the inputs
        assert len(input_shape) == 1
        input_samples = np.vstack(input_samples)
        input_scaler = StandardScaler()

        input_scaler.fit(input_samples)

        output_scaler = None
        num_output_features = len(output_samples[0])
        if self.output_type == OutputType.REGRESSION:
            output_scaler = StandardScaler()
            output_scaler.fit(output_samples)

        # Make the label maps for classification problems
        label_map: Dict[Any, int] = dict()
        reverse_label_map: Dict[int, Any] = dict()
        if self.output_type == OutputType.MULTI_CLASSIFICATION:
            for index, label in enumerate(sorted(unique_labels)):
                label_map[label] = index
                reverse_label_map[index] = label

        self.metadata[INPUT_SCALER] = input_scaler
        self.metadata[OUTPUT_SCALER] = output_scaler
        self.metadata[INPUT_SHAPE] = input_shape
        self.metadata[NUM_OUTPUT_FEATURES] = num_output_features
        self.metadata[SEQ_LENGTH] = seq_length
        self.metadata[INPUT_NOISE] = self.hypers.input_noise

        # Metadata for multiclass classification problems
        self.metadata[NUM_CLASSES] = len(label_map)
        self.metadata[LABEL_MAP] = label_map
        self.metadata[REV_LABEL_MAP] = reverse_label_map

    def make_placeholders(self, is_frozen: bool):
        """
        Creates placeholders for this model.
        """
        pass

    def make_model(self, is_train: bool):
        """
        Builds the computational graph for this model.
        """
        pass

    def make_loss(self):
        """
        Makes the loss function for this model.
        """
        pass

    def predict(self, dataset: Dataset,
                test_batch_size: Optional[int],
                max_num_batches: Optional[int],
                series: DataSeries = DataSeries.TEST) -> DefaultDict[str, Dict[str, Any]]:
        """
        Execute the model to produce a prediction for the given input sample.

        Args:
            dataset: Dataset object used to create input tensors.
            test_batch_size: Batch size to use during testing
            max_num_batches: Maximum number of batches to perform testing on
            flops_dict: Dictionary of FLOPS for each output operation
            series: Series on which to perform classification. Defaults to the testing set.
        Returns:
            The predicted output produced by the model.
        """
        test_batch_size = test_batch_size if test_batch_size is not None else self.hypers.batch_size
        test_batch_generator = dataset.minibatch_generator(series=series,
                                                           batch_size=test_batch_size,
                                                           metadata=self.metadata,
                                                           should_shuffle=False,
                                                           drop_incomplete_batches=True)

        if self.output_type in (OutputType.BINARY_CLASSIFICATION, OutputType.MULTI_CLASSIFICATION):
            return self.predict_classification(test_batch_generator, test_batch_size, max_num_batches)
        else:  # Regression
            return self.predict_regression(test_batch_generator, test_batch_size, max_num_batches)

    def predict_classification(self, test_batch_generator: Iterable[Any],
                               batch_size: int,
                               max_num_batches: Optional[int]) -> DefaultDict[str, Dict[str, float]]:
        raise NotImplementedError()

    def predict_regression(self, test_batch_generator: Iterable[Any],
                           batch_size: int,
                           max_num_batches: Optional[int]) -> DefaultDict[str, Dict[str, float]]:
        raise NotImplementedError()

    def batch_to_feed_dict(self, batch: Dict[str, np.ndarray], is_train: bool, epoch_num: int) -> Dict[tf.Tensor, np.ndarray]:
        """
        Converts the batch value dictionary into a tensorflow feed dictionary.

        Args:
            batch: Batch dictionary as produced by a dataset batch generator.
            is_train: Whether the model is created during training.
            epoch_num: The epoch number.
        Returns:
            A feed dictionary to provide to Tensorflow.
        """
        pass

    def init(self):
        """
        Initializes all variables in the computation graph.
        """
        with self.sess.graph.as_default():
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

    def count_parameters(self) -> int:
        """
        Returns the number of trainable parameters in this model
        """
        num_parameters = 0
        for var in self.trainable_vars:
            num_parameters += np.prod(var.shape)
        return int(num_parameters)

    def make(self, is_train: bool, is_frozen: bool):
        """
        Creates model and optimizer op.

        Args:
            is_train: Whether the model is built for training or just for inference.
            is_frozen: Whether the mode ls built with frozen inputs.
        """
        if self.is_made:
            return  # Prevent building twice

        with self.sess.graph.as_default():
            self.make_placeholders(is_frozen=is_frozen)
            self.make_model(is_train=is_train)

            # Create the global step variable for learning rate decay
            self._global_step = tf.Variable(0, trainable=False)

            # Create the gradient descent optimizer
            self._optimizer = get_optimizer(name=self.hypers.optimizer,
                                            learning_rate=self.hypers.learning_rate,
                                            learning_rate_decay=self.hypers.learning_rate_decay,
                                            global_step=self._global_step,
                                            decay_steps=self.hypers.decay_steps)

            # The loss and optimization criteria are only
            # guaranteed to be defined when the model is built for training
            if is_train:
                self.make_loss()
                self.make_training_step()

        self._is_made = True

    def make_training_step(self):
        """
        Creates the training step for this model. Gradients are clipped
        for stability.
        """
        trainable_vars = self.trainable_vars

        # Compute the gradients
        gradients = tf.gradients(self._ops[self.loss_op_name], trainable_vars)

        # Clip Gradients
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.hypers.gradient_clip)

        # Prune None values from the set of gradients and apply gradient weights
        pruned_gradients = [(grad, var) for grad, var in zip(clipped_gradients, trainable_vars) if grad is not None]

        # Apply clipped gradients
        optimizer_op = self._optimizer.apply_gradients(pruned_gradients)

        # Increment global step counter
        global_step_op = tf.assign_add(self._global_step, 1)

        # Add operations. By coupling the optimizer and the global step ops, we don't need
        # to worry about applying these operations separately.
        self._ops[self.optimizer_op_name] = tf.group(optimizer_op, global_step_op)

    def execute(self, feed_dict: Dict[tf.Tensor, List[Any]], ops: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Executes the model using the given feed dictionary. An optional set of operation names
        may be supplied to prevent executing ALL operations (the default behavior).

        Args:
            feed_dict: The feed dict of input data to pass to Tensorflow.
            ops: The operations to execute. Default behavior is the execute all operations.
        Returns:
            The outputs of the model.
        """
        ops_to_run = self._ops
        if ops is not None:
            ops_to_run = {op_name: op_val for op_name, op_val in self._ops.items() if op_name in ops}

        with self._sess.graph.as_default():
            op_results = self._sess.run(ops_to_run, feed_dict=feed_dict)
            return op_results

    def freeze(self):
        """
        Freezes the Tensorflow computation graph by converting all variables to constants.
        """
        # We need to convert the high-level output names to the corresponding Tensorflow nodes
        # This operation is done by (1) getting output variables and (2) finding the nodes
        # for which these variables are the outputs

        output_nodes: List[str] = []
        for op in self.sess.graph.get_operations():
            for output_name in map(lambda t: t.name, op.outputs):
                if output_name == self.output_op_name:
                    output_nodes.append(op.name)

        # Freeze the corresponding graph
        with self.sess.graph.as_default():
            tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph.as_graph_def(), output_nodes)

    def train(self, dataset: Dataset, should_print: bool, drop_incomplete_batches: bool = False) -> str:
        """
        Trains the model on the given dataset.

        Args:
            dataset: Dataset object containing training, validation and testing partitions
            should_print: Whether we should print results to stdout
            drop_incomplete_minibatches: Whether to drop incomplete batches
        Returns:
            The name of the training run. Training results are logged to a pickle file with the name
            model-train-log_{name}.pkl.gz.
        """
        self.load_metadata(dataset)

        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        name = NAME_FMT.format(self.name, dataset.dataset_name, current_time)

        # Make Model and Initialize variables
        self.make(is_train=True, is_frozen=False)
        self.init()

        print('Created model with {0} trainable parameters.'.format(self.count_parameters()))

        # Variables for early stopping
        best_valid_metric = 0.0 if is_classification(self.output_type) else BIG_NUMBER
        num_not_improved = 0

        # Lists for logging training results
        loss_dict: DefaultDict[str, List[float]] = defaultdict(list)
        acc_dict: DefaultDict[str, List[float]] = defaultdict(list)

        # Execute training and validation epochs
        for epoch in range(self.hypers.epochs):
            if should_print:
                print('-------- Epoch {0} --------'.format(epoch))

            train_generator = dataset.minibatch_generator(DataSeries.TRAIN,
                                                          batch_size=self.hypers.batch_size,
                                                          metadata=self.metadata,
                                                          should_shuffle=True,
                                                          drop_incomplete_batches=drop_incomplete_batches)

            # Collect the training operations. For classification tasks, we also include the accuracy.
            train_ops = [self.loss_op_name, self.optimizer_op_name, self.accuracy_op_name]

            train_accuracy = 0.0
            train_loss = 0.0
            train_samples = 0

            for batch_idx, batch in enumerate(train_generator):
                feed_dict = self.batch_to_feed_dict(batch, is_train=True, epoch_num=epoch)

                batch_size = len(batch[OUTPUT])

                # Run the training operations
                train_results = self.execute(feed_dict, train_ops)

                # Aggregate the loss and average accuracy
                train_loss += train_results[self.loss_op_name] * batch_size
                train_accuracy += train_results.get(self.accuracy_op_name, 0.0) * batch_size
                train_samples += batch_size

                avg_loss_so_far = train_loss / train_samples
                avg_acc_so_far = train_accuracy / train_samples

                if should_print:
                    if is_classification(self.output_type):
                        print('Train Batch {0}. Avg loss so far: {1:.4f}, Avg accuracy so far: {2:.4f}'.format(batch_idx, avg_loss_so_far, avg_acc_so_far), end='\r')
                    else:
                        print('Train Batch {0}. Avg loss so far: {1:.4f}'.format(batch_idx, avg_loss_so_far), end='\r')

            if should_print:
                print()  # Clear the line

            avg_train_loss = train_loss / train_samples
            avg_train_acc = train_accuracy / train_samples

            loss_dict[TRAIN].append(avg_train_loss)
            acc_dict[TRAIN].append(avg_train_acc)

            valid_generator = dataset.minibatch_generator(DataSeries.VALID,
                                                          batch_size=self.hypers.batch_size,
                                                          metadata=self.metadata,
                                                          should_shuffle=True,
                                                          drop_incomplete_batches=drop_incomplete_batches)

            # Collect the training operations. For classification tasks, we also include the accuracy.
            valid_ops = [self.loss_op_name, self.accuracy_op_name]

            valid_accuracy = 0.0
            valid_loss = 0.0
            valid_samples = 0

            for batch_idx, batch in enumerate(valid_generator):
                feed_dict = self.batch_to_feed_dict(batch, is_train=False, epoch_num=epoch)

                batch_size = len(batch[OUTPUT])

                # Run the training operations
                valid_results = self.execute(feed_dict, valid_ops)

                # Aggregate the loss and average accuracy
                valid_loss += valid_results[self.loss_op_name] * batch_size
                valid_accuracy += valid_results.get(self.accuracy_op_name, 0.0) * batch_size
                valid_samples += batch_size

                avg_loss_so_far = valid_loss / valid_samples
                avg_acc_so_far = valid_accuracy / valid_samples

                if should_print:
                    if is_classification(self.output_type):
                        print('Validation Batch {0}. Avg loss so far: {1:.4f}, Avg accuracy so far: {2:.4f}'.format(batch_idx, avg_loss_so_far, avg_acc_so_far), end='\r')
                    else:
                        print('Validation Batch {0}. Avg loss so far: {1:.4f}'.format(batch_idx, avg_loss_so_far), end='\r')

            if should_print:
                print()  # Clear the line

            avg_valid_loss = valid_loss / valid_samples
            avg_valid_accuracy = valid_accuracy / valid_samples

            loss_dict[VALID].append(avg_valid_loss)
            acc_dict[VALID].append(avg_valid_accuracy)

            # Detect improvement using the validation set
            has_improved = False
            if is_classification(self.output_type):
                if avg_valid_accuracy > best_valid_metric:
                    best_valid_metric = avg_valid_accuracy
                    has_improved = True
            else:
                if avg_valid_loss < best_valid_metric:
                    best_valid_metric = avg_valid_loss
                    has_improved = True

            # Save the model upon improvement
            if has_improved:
                if should_print:
                    print('Saving model...')

                self.save(name=name, data_folders=dataset.data_folders)
                num_not_improved = 0
            else:
                num_not_improved += 1

            if num_not_improved >= self.hypers.patience:
                if should_print:
                    print('Terminating due to early stopping.')
                break

        # Log the ending time. This tracks the time to train this model
        ending_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        # Save training metrics
        metrics_dict = dict(loss=loss_dict, accuracy=acc_dict, start_time=current_time, end_time=ending_time)
        log_file = os.path.join(self.save_folder, TRAIN_LOG_PATH.format(name))
        save_by_file_suffix(metrics_dict, log_file)

        return name

    def save(self, name: str, data_folders: Dict[DataSeries, str]):
        """
        Save model weights, hyper-parameters, and metadata

        Args:
            name: Name of the model
            data_folders: Data folders used for training and validation
        """
        # Save hyperparameters
        params_path = os.path.join(self.save_folder, HYPERS_PATH.format(name))
        save_by_file_suffix(self.hypers.as_dict(), params_path)

        # Save metadata
        data_folders_dict = {series.name: path for series, path in data_folders.items()}
        metadata_path = os.path.join(self.save_folder, METADATA_PATH.format(name))
        save_by_file_suffix(dict(metadata=self.metadata, data_folders=data_folders_dict), metadata_path)

        with self.sess.graph.as_default():
            model_path = os.path.join(self.save_folder, MODEL_PATH.format(name))

            # Get all variable values
            trainable_vars = self.trainable_vars
            vars_to_save = {var.name: var for var in trainable_vars}
            vars_dict = self.sess.run(vars_to_save)

            # Save results
            save_by_file_suffix(vars_dict, model_path)

    def restore(self, name: str, is_train: bool, is_frozen: bool):
        """
        Restore model metadata, hyper-parameters, and trainable parameters.
        """
        # Restore hyperparameters
        params_path = os.path.join(self.save_folder, HYPERS_PATH.format(name))
        self.hypers = HyperParameters.create_from_file(params_path)

        # Restore metadata
        metadata_path = os.path.join(self.save_folder, METADATA_PATH.format(name))
        train_metadata = read_by_file_suffix(metadata_path)
        self.metadata = train_metadata['metadata']

        # Build the model
        self.make(is_train=is_train, is_frozen=is_frozen)

        # Initialize all variables (some may not be trainable)
        self.init()

        # Restore the trainable parameters
        with self.sess.graph.as_default():
            model_path = os.path.join(self.save_folder, MODEL_PATH.format(name))
            vars_dict = read_by_file_suffix(model_path)

            # Collect all saved variables
            assign_ops = []
            for trainable_var in self.trainable_vars:
                saved_value = vars_dict.get(trainable_var.name)
                if saved_value is None:
                    print('WARNING: No value for {0}'.format(trainable_var.name))
                else:
                    assign_op = trainable_var.assign(saved_value, use_locking=True, read_value=False)
                    assign_ops.append(assign_op)

            # Execute assignment
            self.sess.run(assign_ops)

        if is_frozen:
            self.freeze()
