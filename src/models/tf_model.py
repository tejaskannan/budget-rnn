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
from layers.output_layers import OutputType
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

        self._optimizers: Dict[str, tf.train.Optimizer] = dict()  # Map from optimizer op name to optimizer instance
        self._ops: Dict[str, tf.Tensor] = dict()
        self._placeholders: Dict[str, tf.Tensor] = dict()
        self._global_steps: Dict[str, tf.Variable] = dict()  # Map from op name to counter for weight decay
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
    def optimizer_op_names(self) -> List[str]:
        return [OPTIMIZER_OP]

    @property
    def loss_op_names(self) -> List[str]:
        return [LOSS]

    @property
    def accuracy_op_names(self) -> List[str]:
        return [ACCURACY]

    @property
    def f1_op_names(self) -> List[str]:
        return [F1_SCORE]

    @property
    def output_ops(self) -> List[str]:
        raise NotImplementedError()

    @property
    def global_step_op_names(self) -> List[str]:
        return [GLOBAL_STEP]

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

        input_scaler = None
        if self.hypers.model_params['normalize_inputs']:
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

        n_global_steps, n_optimizer_ops = len(self.global_step_op_names), len(self.optimizer_op_names)
        assert n_global_steps == n_optimizer_ops, 'Must have the same number of optimizer ops ({0}) as global step ops ({1}).'.format(n_optimizer_ops, n_global_steps)

        with self.sess.graph.as_default():
            self.make_placeholders(is_frozen=is_frozen)
            self.make_model(is_train=is_train)

            # self._global_step = tf.Variable(0, trainable=False)

            for global_step_name, optimizer_op_name in zip(self.global_step_op_names, self.optimizer_op_names):
                self._global_steps[global_step_name] = tf.Variable(0, trainable=False)

                self._optimizers[optimizer_op_name] = get_optimizer(name=self.hypers.optimizer,
                                                                    learning_rate=self.hypers.learning_rate,
                                                                    learning_rate_decay=self.hypers.learning_rate_decay,
                                                                    global_step=self._global_steps[global_step_name],
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
        for better numerical stability.

        Args:
            loss_ops: Optional dictionary mapping loss operations to a list of trainable variables.
                The learning rates can be individually scaled for each variable.
        """
        assert len(self.optimizer_op_names) == len(self.loss_op_names), 'Must have the same number of loss ops ({0}) as optimizer ops ({1])'.format(len(self.loss_op_names), len(self.optimizer_op_names))
        optimizer_ops = []
        trainable_vars = self.trainable_vars

        # Compute gradients for each loss operation
        for loss_op, optimizer_op_name, global_step_name in zip(self.loss_op_names, self.optimizer_op_names, self.global_step_op_names):
            gradients = tf.gradients(self._ops[loss_op], trainable_vars)

            # Clip Gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.hypers.gradient_clip)

            # Prune None values from the set of gradients and apply gradient weights
            pruned_gradients = [(grad, var) for grad, var in zip(clipped_gradients, trainable_vars) if grad is not None]

            # Apply clipped gradients
            optimizer_op = self._optimizers[optimizer_op_name].apply_gradients(pruned_gradients)

            # Increment global step counter
            global_step_op = tf.assign_add(self._global_steps[global_step_name], 1)

            # Add operations. By coupling the optimizer and the global step ops, we don't need
            # to worry about applying these operations separately.
            self._ops[optimizer_op_name] = tf.group(optimizer_op, global_step_op)

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

    def freeze(self, outputs: List[str]):
        """
        Freezes the Tensorflow computation graph by converting all variables to constants.

        Args:
            outputs: List of high-level operations representing the model outputs
        """
        # We need to convert the high-level output names to the corresponding Tensorflow nodes
        # This operation is done by (1) getting output variables and (2) finding the nodes
        # for which these variables are the outputs
        output_names = [self.ops[op].name for op in outputs]

        output_nodes: List[str] = []
        for op in self.sess.graph.get_operations():
            for output_name in map(lambda t: t.name, op.outputs):
                if output_name in output_names:
                    output_nodes.append(op.name)

        # Freeze the corresponding graph
        with self.sess.graph.as_default():
            tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph.as_graph_def(), output_nodes)

    def train(self, dataset: Dataset, drop_incomplete_batches: bool = False) -> str:
        """
        Trains the model on the given dataset.

        Args:
            dataset: Dataset object containing training, validation and testing partitions
            drop_incomplete_minibatches: Whether to drop incomplete batches
        Returns:
            The name of the training run. Training results are logged to a pickle file with the name
            model-train-log_{name}.pkl.gz.
        """
        self.load_metadata(dataset)

        current_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        name = NAME_FMT.format(self.name, dataset.dataset_name, current_date)

        # Make Model and Initialize variables
        self.make(is_train=True, is_frozen=False)
        self.init()

        print(f'Created model with {self.count_parameters()} trainable parameters.')

        loss_dict: DefaultDict[str, List[float]] = defaultdict(list)
        acc_dict: DefaultDict[str, List[float]] = defaultdict(list)

        # Initialize dictionary to hold metrics to measure improvement
        init_metric_value = BIG_NUMBER if self.output_type == OutputType.REGRESSION else -BIG_NUMBER
        best_valid_metric_dict: DefaultDict[str, float] = defaultdict(lambda: init_metric_value)

        # Create dictionary to map accuracy operations to loss operations
        accuracy_loss_dict: Dict[str, str] = dict()
        assert len(self.accuracy_op_names) == len(self.loss_op_names) or len(self.loss_op_names) == 1, f'Misaligned accuracy and loss operations.'
        if len(self.loss_op_names) == 1:
            accuracy_loss_dict = {acc_op: self.loss_op_names[0] for acc_op in self.accuracy_op_names}
        else:
            accuracy_loss_dict = {acc_op: loss_op for acc_op, loss_op in zip(self.accuracy_op_names, self.loss_op_names)}

        # Create dictionary mapping each loss operation to the set of corresponding trainable variables. This mapping does not change
        # after graph construction, so we create if before starting training.
        loss_var_dict: Dict[str, tf.Variable] = dict()
        with self.sess.graph.as_default():
            for loss_op in self.loss_op_names:
                loss_var_dict[loss_op] = [v.name for v in variables_for_loss_op(self.trainable_vars, self.ops[loss_op])]

        # Dictionary to track whether to continue training variables for each loss operations
        num_not_improved: Dict[str, int] = {loss_op: 0 for loss_op in self.loss_op_names}
        has_stopped: Dict[str, bool] = {loss_op: False for loss_op in self.loss_op_names}

        # Execute training and validation epochs
        for epoch in range(self.hypers.epochs):
            print(f'-------- Epoch {epoch} --------')

            train_generator = dataset.minibatch_generator(DataSeries.TRAIN,
                                                          batch_size=self.hypers.batch_size,
                                                          metadata=self.metadata,
                                                          should_shuffle=True,
                                                          drop_incomplete_batches=drop_incomplete_batches)

            epoch_train_loss: DefaultDict[str, float] = defaultdict(float)
            epoch_train_acc: DefaultDict[str, float] = defaultdict(float)

            # Collect the training operations
            train_ops_to_run = list(optimizer_op for optimizer_op, loss_op in zip(self.optimizer_op_names, self.loss_op_names) if not has_stopped[loss_op])
            train_ops_to_run += list(loss_op for loss_op, stopped_status in has_stopped.items() if not stopped_status)

            if self.output_type in (OutputType.BINARY_CLASSIFICATION, OutputType.MULTI_CLASSIFICATION):
                train_ops_to_run += self.accuracy_op_names

            train_batch_counter = 1
            for batch in train_generator:
                feed_dict = self.batch_to_feed_dict(batch, is_train=True, epoch_num=epoch)

                # Run the training operations
                train_results = self.execute(feed_dict, train_ops_to_run)

                batch_loss = 0.0
                for loss_op_name in self.loss_op_names:
                    if loss_op_name in train_results:
                        avg_batch_loss = np.average(train_results[loss_op_name])
                        batch_loss += avg_batch_loss
                        epoch_train_loss[loss_op_name] += avg_batch_loss

                train_loss_agg = np.average(list(epoch_train_loss.values()))
                avg_train_loss_so_far = train_loss_agg / train_batch_counter

                for acc_op_name in self.accuracy_op_names:
                    epoch_train_acc[acc_op_name] += train_results.get(acc_op_name, 0.0)

                train_acc_agg = 0.0
                train_acc_values = list(epoch_train_acc.values())
                if len(train_acc_values) > 0:
                    train_acc_agg = np.average(train_acc_values)

                avg_train_acc_so_far = train_acc_agg / train_batch_counter

                if self.output_type in (OutputType.BINARY_CLASSIFICATION, OutputType.MULTI_CLASSIFICATION):
                    print(f'Train Batch {train_batch_counter}. Avg loss so far: {avg_train_loss_so_far:.4f}, Avg accuracy so far: {avg_train_acc_so_far:.4f}', end='\r')
                else:
                    print(f'Train Batch {train_batch_counter}. Avg loss so far: {avg_train_loss_so_far:.4f}', end='\r')

                train_batch_counter += 1
            print()

            # Perform validation
            valid_generator = dataset.minibatch_generator(DataSeries.VALID,
                                                          batch_size=self.hypers.batch_size,
                                                          metadata=self.metadata,
                                                          should_shuffle=False,
                                                          drop_incomplete_batches=drop_incomplete_batches)

            epoch_valid_loss: DefaultDict[str, float] = defaultdict(float)  # Map from loss_op -> average epoch loss
            epoch_valid_acc: DefaultDict[str, float] = defaultdict(float)  # Map from accuracy op -> epoch accuracy

            # Collect the validation ops
            valid_ops_to_run = list(loss_op for loss_op, stopped_status in has_stopped.items() if not stopped_status)
            if self.output_type in (OutputType.BINARY_CLASSIFICATION, OutputType.MULTI_CLASSIFICATION):
                valid_ops_to_run += self.accuracy_op_names

            valid_batch_counter = 1
            for batch in valid_generator:
                feed_dict = self.batch_to_feed_dict(batch, is_train=False, epoch_num=epoch)

                # Run the validation operations
                valid_results = self.execute(feed_dict, valid_ops_to_run)

                batch_loss = 0.0
                for loss_op_name in self.loss_op_names:
                    if loss_op_name in valid_results:
                        avg_batch_loss = np.average(valid_results[loss_op_name])
                        batch_loss += avg_batch_loss
                        epoch_valid_loss[loss_op_name] += avg_batch_loss

                valid_loss_agg = np.average(list(epoch_valid_loss.values()))
                avg_valid_loss_so_far = valid_loss_agg / valid_batch_counter

                # Compute accuracy
                for acc_op_name in self.accuracy_op_names:
                    epoch_valid_acc[acc_op_name] += valid_results.get(acc_op_name, 0.0)

                valid_acc_agg = 0.0
                valid_acc_values = list(epoch_valid_acc.values())
                if len(valid_acc_values) > 0:
                    valid_acc_agg = np.average(valid_acc_values)

                avg_valid_acc_so_far = valid_acc_agg / valid_batch_counter

                if self.output_type in (OutputType.BINARY_CLASSIFICATION, OutputType.MULTI_CLASSIFICATION):
                    print(f'Valid Batch {valid_batch_counter}. Avg loss so far: {avg_valid_loss_so_far:.4f}, Avg accuracy so far: {avg_valid_acc_so_far:.4f}', end='\r')
                else:
                    print(f'Valid Batch {valid_batch_counter}. Avg loss so far: {avg_valid_loss_so_far:.4f}', end='\r')

                valid_batch_counter += 1
            print()

            # Log train and validation metrics for each epoch
            loss_dict[TRAIN].append(np.average(list(epoch_train_loss.values())) / train_batch_counter)
            loss_dict[VALID].append(np.average(list(epoch_valid_loss.values())) / valid_batch_counter)
            acc_dict[TRAIN].append(np.average(list(epoch_train_acc.values())) / train_batch_counter)
            acc_dict[VALID].append(np.average(list(epoch_valid_acc.values())) / valid_batch_counter)

            # Collect loss operation to save
            metric_ops = self.loss_op_names if self.output_type == OutputType.REGRESSION else self.accuracy_op_names
            has_improved: Dict[str, bool] = {loss_op: False for loss_op in self.loss_op_names}
            loss_ops_to_save: Set[str] = set()

            # When there is a single loss op, we always use the average loss / accuracy
            if len(self.loss_op_names) == 1:
                loss_op_name = self.loss_op_names[0]

                # For regression tasks, we want to minimize loss
                if self.output_type == OutputType.REGRESSION:
                    valid_loss = np.average(list(epoch_valid_loss.values()))
                    if valid_loss < best_valid_metric_dict[loss_op_name]:
                        loss_ops_to_save.add(loss_op_name)
                        best_valid_metric_dict[loss_op_name] = valid_loss
                        has_improved[loss_op_name] = True
                else:
                    # For classification tasks, we want to maximize accuracy
                    valid_acc = np.average(list(epoch_valid_acc.values()))
                    if valid_acc > best_valid_metric_dict[ACCURACY]:
                        loss_ops_to_save.add(loss_op_name)
                        best_valid_metric_dict[ACCURACY] = valid_acc
                        has_improved[loss_op_name] = True
            else:
                # For some models, we have many independent loss operations. This helps train multiple models in parallel.
                # In this case, we treat each loss operation as a separate model for early stopping.
                for op_name in metric_ops:
                    # For regression tasks, we want to minimize loss
                    if self.output_type == OutputType.REGRESSION:
                        valid_loss = epoch_valid_loss[op_name]
                        if valid_loss < best_valid_metric_dict[op_name]:
                            loss_ops_to_save.add(op_name)
                            best_valid_metric_dict[op_name] = valid_loss
                            has_improved[op_name] = True

                    # For classification tasks, we want to maximize accuracy
                    if self.output_type in (OutputType.BINARY_CLASSIFICATION, OutputType.MULTI_CLASSIFICATION):
                        valid_acc = epoch_valid_acc[op_name]

                        if valid_acc > best_valid_metric_dict[op_name]:
                            # Save the corresponding loss operation
                            loss_op_name = accuracy_loss_dict[op_name]
                            loss_ops_to_save.add(loss_op_name)

                            best_valid_metric_dict[op_name] = valid_acc
                            has_improved[loss_op_name] = True

            # Save model if necessary
            loss_ops_list = list(sorted(loss_ops_to_save))
            if len(loss_ops_list) > 0:
                print('Saving model for operations: {0}'.format(','.join(loss_ops_list)))
                self.save(name=name, data_folders=dataset.data_folders, loss_ops=loss_ops_list, loss_var_dict=loss_var_dict)

            # Increment the improvement counter
            for loss_op, improved_status in has_improved.items():
                if improved_status:
                    num_not_improved[loss_op] = 0
                else:
                    num_not_improved[loss_op] += 1

                if num_not_improved[loss_op] >= self.hypers.patience:
                    has_stopped[loss_op] = True

            # Exit of all loss operations have stopped
            if all(has_stopped.values()):
                print('Exiting due to Early Stopping')
                break

            # Call garbage collector at the end of each iteration (just to be safe)
            gc.collect()

        # Save training metrics
        metrics_dict = dict(loss=loss_dict, accuracy=acc_dict)
        log_file = os.path.join(self.save_folder, TRAIN_LOG_PATH.format(name))
        save_by_file_suffix(metrics_dict, log_file)

        return name

    def save(self, name: str, data_folders: Dict[DataSeries, str], loss_ops: Optional[List[str]], loss_var_dict: Dict[str, List[str]]):
        """
        Save model weights, hyper-parameters, and metadata

        Args:
            name: Name of the model
            data_folders: Data folders used for training and validation
            loss_ops: Loss operations for which to save variables. None value indicates that ALL variables
                are to be saved
        """
        # Save hyperparameters
        params_path = os.path.join(self.save_folder, HYPERS_PATH.format(name))
        save_by_file_suffix(self.hypers.__dict__(), params_path)

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

            # Save only variables with existing gradients
            if loss_ops is not None:
                # Load existing variables
                saved_vars: Dict[str, np.ndarray] = dict()
                if os.path.exists(model_path):
                    saved_vars = read_by_file_suffix(model_path)

                # Get variables to save
                for loss_op in loss_ops:
                    # Fetch the corresponding trainable variables
                    valid_vars = loss_var_dict[loss_op]
                    valid_vars_dict = {v: vars_dict[v] for v in valid_vars}

                    # Update values for the valid variables
                    saved_vars.update(valid_vars_dict)

                # Write the updated dictionary
                vars_dict = saved_vars

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
            self.freeze(self.output_ops)
