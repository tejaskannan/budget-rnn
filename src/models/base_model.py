import tensorflow as tf
import numpy as np
import re
from os import listdir
from dpu_utils.utils import RichPath
from datetime import datetime
from collections import defaultdict
from typing import Optional, Iterable, Dict, Any, Union, List, DefaultDict

from dataset.dataset import Dataset, DataSeries
from layers.output_layers import OutputType
from utils.hyperparameters import HyperParameters
from utils.tfutils import get_optimizer, variables_for_loss_op
from utils.file_utils import to_rich_path
from utils.constants import BIG_NUMBER


class Model:

    def __init__(self, hyper_parameters: HyperParameters, save_folder: Union[str, RichPath]):
        self.hypers = hyper_parameters
        self.save_folder = to_rich_path(save_folder)
        self.metadata: Dict[str, Any] = dict()

        self._sess = tf.Session(graph=tf.Graph())
        self._optimizer = get_optimizer(self.hypers.optimizer, self.hypers.learning_rate)
        self._ops: Dict[str, tf.Tensor] = dict()
        self._placeholders: Dict[str, tf.Tensor] = dict()
        self._loss_ops: Dict[str, List[VariableWithWeight]] = None

        # Dictionary with inference operations
        self._inference_ops: Dict[str, tf.Tensor] = dict()

        # Make the output folder
        self.save_folder.make_as_dir()
        self.name = ''

    @property
    def ops(self) -> Dict[str, tf.Tensor]:
        return self._ops

    @property
    def placeholders(self) -> Dict[str, tf.Tensor]:
        return self._placeholders
    
    @property
    def optimizer_op_name(self) -> str:
        return 'optimizer_op'

    @property
    def loss_op_names(self) -> List[str]:
        return ['loss']

    @property
    def accuracy_op_names(self) -> List[str]:
        return ['accuracy']

    @property
    def sess(self) -> tf.Session:
        return self._sess

    @property
    def trainable_vars(self) -> List[tf.Variable]:
        return list(self.sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

    def load_metadata(self, dataset: Dataset):
        """
        Loads metadata from the dataset. For example, this function
        may construct the token vocabulary. Results are stored
        directly into self.metadata.
        """
        pass

    def make_placeholders(self):
        """
        Creates placeholders for this model.
        """
        pass

    def make_model(self):
        """
        Builds the computational graph for this model.
        """
        pass

    def make_loss(self):
        """
        Makes the loss function for this model.
        """
        pass

    def predict(self, samples: List[Dict[str, Any]], dataset: Dataset) -> Dict[Any, Any]:
        """
        Execute the model to produce a prediction for the given input sample.

        Args:
            samples: Input samples to predict outputs for.
            dataset: Dataset object used to create input tensors.
        Returns:
            The predicted output produced by the model.
        """
        pass

    def batch_to_feed_dict(self, batch: Dict[str, np.ndarray], is_train: bool) -> Dict[tf.Tensor, np.ndarray]:
        """
        Converts the batch value dictionary into a tensorflow feed dictionary.

        Args:
            batch: Batch dictionary as produced by a dataset batch generator.
            is_train: Whether the model is created during training.
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

    def make(self, is_train: bool):
        """
        Creates model and optimizer op.

        Args:
            is_train: Whether the model is built for training or just for inference.
        """
        with self._sess.graph.as_default():
            self.make_placeholders()
            self.make_model(is_train=is_train)

            # The loss and optimization criteria are only
            # guaranteed to be defined when the model is built for training
            if is_train:
                self.make_loss()
                self.make_training_step()

    def make_training_step(self):
        """
        Creates the training step for this model. Gradients are clipped
        for better numerical stability.

        Args:
            loss_ops: Optional dictionary mapping loss operations to a list of trainable variables.
                The learning rates can be individually scaled for each variable.
        """
        optimizer_ops = []
        trainable_vars = self.trainable_vars
       
        # Compute gradients for each loss operation
        for loss_op in self.loss_op_names:
            gradients = tf.gradients(self._ops[loss_op], trainable_vars)

            # Clip Gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.hypers.gradient_clip)

            # Prune None values from the set of gradients and apply gradient weights
            pruned_gradients = [(grad, var) for grad, var in zip(clipped_gradients, trainable_vars) if grad is not None]

            # Apply clipped gradients
            optimizer_op = self._optimizer.apply_gradients(pruned_gradients)
            optimizer_ops.append(optimizer_op)

        # Group all optimizer operations
        self._ops[self.optimizer_op_name] = tf.group(optimizer_ops)

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
        name = f'{dataset.dataset_name}-{current_date}_model_best'
        if len(self.name) > 0:
            name = f'{self.name}-{name}'

        # Make Model and Initialize variables
        self.make(is_train=True)
        self.init()

        print(f'Created model with {self.count_parameters()} trainable parameters.')

        train_loss_dict: DefaultDict[str, List[float]] = defaultdict(list)
        valid_loss_dict: DefaultDict[str, List[float]] = defaultdict(list)
        best_valid_loss_dict: DefaultDict[str, float] = defaultdict(lambda: BIG_NUMBER)

        num_not_improved = 0
        for epoch in range(self.hypers.epochs):
            print(f'-------- Epoch {epoch} --------')

            train_generator = dataset.minibatch_generator(DataSeries.TRAIN,
                                                          batch_size=self.hypers.batch_size,
                                                          metadata=self.metadata,
                                                          should_shuffle=True,
                                                          drop_incomplete_batches=drop_incomplete_batches)
            
            epoch_train_loss: DefaultDict[str, float] = defaultdict(float)
            epoch_train_acc: DefaultDict[str, float] = defaultdict(float)

            for i, batch in enumerate(train_generator):
                feed_dict = self.batch_to_feed_dict(batch, is_train=True)
                ops_to_run = [self.optimizer_op_name] + self.loss_op_names
                
                if self.output_type == OutputType.CLASSIFICATION:
                    ops_to_run += self.accuracy_op_names

                train_results = self.execute(feed_dict, ops_to_run)

                batch_loss = 0.0
                for loss_op_name in self.loss_op_names:
                    avg_batch_loss = np.average(train_results[loss_op_name])
                    batch_loss += avg_batch_loss
                    train_loss_dict[loss_op_name].append(avg_batch_loss)
                    epoch_train_loss[loss_op_name] += avg_batch_loss

                train_loss_agg = np.average(list(epoch_train_loss.values()))
                avg_train_loss_so_far = train_loss_agg / (i+1)

                for acc_op_name in self.accuracy_op_names:
                    epoch_train_acc[acc_op_name] += train_results[acc_op_name]

                train_acc_agg = np.average(list(epoch_train_acc.values()))
                avg_train_acc_so_far = train_acc_agg / (i+1)

                if self.hypers.model_params.get('output_type', '').lower() == 'classification':
                    print(f'Train Batch {i}. Avg loss so far: {avg_train_loss_so_far:.4f}, Avg accuracy so far: {avg_train_acc_so_far:.4f}', end='\r')                
                else:
                    print(f'Train Batch {i}. Avg loss so far: {avg_train_loss_so_far:.4f}', end='\r')

            print()

            valid_generator = dataset.minibatch_generator(DataSeries.VALID,
                                                          batch_size=self.hypers.batch_size,
                                                          metadata=self.metadata,
                                                          should_shuffle=False,
                                                          drop_incomplete_batches=drop_incomplete_batches)
            
            epoch_valid_loss: DefaultDict[str, float] = defaultdict(float)
            epoch_valid_acc: DefaultDict[str, float] = defaultdict(float)
            
            for i, batch in enumerate(valid_generator):
                feed_dict = self.batch_to_feed_dict(batch, is_train=False)

                ops_to_run: List[str] = []
                if self.output_type == OutputType.CLASSIFICATION:
                    ops_to_run += self.accuracy_op_names
                
                ops_to_run += self.loss_op_names
                valid_results = self.execute(feed_dict, ops_to_run)

                batch_loss = 0.0
                for loss_op_name in self.loss_op_names:
                    avg_batch_loss = np.average(valid_results[loss_op_name])
                    batch_loss += avg_batch_loss
                    valid_loss_dict[loss_op_name].append(avg_batch_loss)
                    epoch_valid_loss[loss_op_name] += avg_batch_loss

                valid_loss_agg = np.average(list(epoch_valid_loss.values()))
                avg_valid_loss_so_far = valid_loss_agg / (i+1)

                for acc_op_name in self.accuracy_op_names:
                    epoch_valid_acc[acc_op_name] += valid_results[acc_op_name]

                valid_acc_agg = np.average(list(epoch_valid_acc.values()))
                avg_valid_acc_so_far = valid_acc_agg / (i+1)

                if self.hypers.model_params.get('output_type', '').lower() == 'classification':
                    print(f'Valid Batch {i}. Avg loss so far: {avg_valid_loss_so_far:.4f}, Avg accuracy so far: {avg_valid_acc_so_far:.4f}', end='\r')
                else:
                    print(f'Valid Batch {i}. Avg loss so far: {avg_valid_loss_so_far:.4f}', end='\r')

            print()

            # Collect operations to save
            has_improved = False
            loss_ops_to_save: List[str] = []
            for loss_op_name, valid_loss in epoch_valid_loss.items():
                if valid_loss < best_valid_loss_dict[loss_op_name]:
                    loss_ops_to_save.append(loss_op_name)
                    best_valid_loss_dict[loss_op_name] = valid_loss
                    has_improved = True

            # Save model if necessary
            if len(loss_ops_to_save) > 0:
                print('Saving model for operations: {0}'.format(','.join(loss_ops_to_save)))
                self.save(name=name, data_folders=dataset.data_folders, loss_ops=loss_ops_to_save)

            if has_improved:
                num_not_improved = 0
            else:
                num_not_improved += 1

            if num_not_improved >= self.hypers.patience:
                print('Exiting due to Early Stopping')
                break

        metrics_dict = dict(train_losses=train_loss_dict, valid_losses=valid_loss_dict)

        log_file = self.save_folder.join(f'model-train-log-{name}.pkl.gz')
        log_file.save_as_compressed_file([metrics_dict])

        return name

    def save(self, name: str, data_folders: Dict[DataSeries, str], loss_ops: Optional[List[str]]):
        """
        Save model weights, hyper-parameters, and metadata

        Args:
            name: Name of the model
            data_folders: Data folders used for training and validation
            loss_ops: Loss operations for which to save variables. None value indicates that ALL variables
                are to be saved
        """
        # Save hyperparameters
        params_path = self.save_folder.join(f'model-hyper-params-{name}.pkl.gz')
        params_path.save_as_compressed_file(self.hypers.__dict__())

        # Save metadata
        data_folders_dict = {series.name: path for series, path in data_folders.items()}
        metadata_path = self.save_folder.join(f'model-metadata-{name}.pkl.gz')
        metadata_path.save_as_compressed_file(dict(metadata=self.metadata, data_folders=data_folders_dict))

        with self.sess.graph.as_default():
            model_path = self.save_folder.join(f'model-{name}.pkl.gz')
            
            # Get all variable values
            trainable_vars = self.trainable_vars
            vars_to_save = {var.name: var for var in trainable_vars}
            vars_dict = self.sess.run(vars_to_save)

            # Save only variables with existing gradients
            if loss_ops is not None:
                # Load existing variables
                saved_vars: Dict[str, np.ndarray] = dict()
                if model_path.exists():
                    saved_vars = model_path.read_by_file_suffix()

                # Get variables to save
                for loss_op in loss_ops:
                    valid_vars = variables_for_loss_op(trainable_vars, self.ops[loss_op])
                    valid_vars_dict = {v.name: vars_dict[v.name] for v in valid_vars}

                    # Update values for the valid variables
                    saved_vars.update(valid_vars_dict)

                # Write the updated dictionary
                vars_dict = saved_vars

            # Save results
            model_path.save_as_compressed_file(vars_dict)

    def restore(self, name: str, is_train: bool):
        """
        Restore model metadata, hyper-parameters, and trainable parameters.
        """
        # Restore hyperparameters
        params_path = self.save_folder.join(f'model-hyper-params-{name}.pkl.gz')
        self.hypers = HyperParameters.create_from_file(params_path)

        # Restore metadata
        metadata_path = self.save_folder.join(f'model-metadata-{name}.pkl.gz')
        train_metadata = metadata_path.read_by_file_suffix()
        self.metadata = train_metadata['metadata']

        # Build the model
        self.make(is_train=is_train)
    
        # Restore the trainable parameters
        with self.sess.graph.as_default():
            model_path = self.save_folder.join(f'model-{name}.pkl.gz')
            vars_dict = model_path.read_by_file_suffix()

            # Collect all saved variables
            assign_ops = []
            for trainable_var in self.trainable_vars:
                saved_value = vars_dict[trainable_var.name]
                assign_op = trainable_var.assign(saved_value, use_locking=True, read_value=False)
                assign_ops.append(assign_op)

            # Execute assignment
            self.sess.run(assign_ops)

