import tensorflow as tf
import numpy as np
import re
from os import listdir
from dpu_utils.utils import RichPath
from datetime import datetime
from collections import defaultdict
from typing import Optional, Iterable, Dict, Any, Union, List, DefaultDict

from dataset.dataset import Dataset, DataSeries
from utils.hyperparameters import HyperParameters, extract_hyperparameters
from utils.tfutils import get_optimizer
from utils.file_utils import to_rich_path
from utils.constants import BIG_NUMBER


class VariableWithWeight:

    __slots__ = ['variable', 'weight']

    def __init__(self, variable: tf.Variable, weight: float = 1.0):
        assert 0.0 <= weight <= 1.0, 'Weight must be a number between 0 and 1.'
        self.variable = variable
        self.weight = weight

    def __str__(self) -> str:
        return f'({self.variable.name}, {self.weight})'


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
    def optimizer_op_name(self) -> str:
        return 'optimizer_op'

    @property
    def loss_op_names(self) -> List[str]:
        return ['loss']

    def get_variable_group(self, loss_op_name: str) -> List[tf.Variable]:
        return [var for var in self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]

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
        with self._sess.graph.as_default():
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

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
                self.make_training_step(loss_ops=self._loss_ops)

    def make_training_step(self, loss_ops: Optional[Dict[str, List[VariableWithWeight]]] = None):
        """
        Creates the training step for this model. Gradients are clipped
        for better numerical stability.

        Args:
            loss_ops: Optional dictionary mapping loss operations to a the set of trainable variables.
                The learning rates can be individually scaled for each variable.
        """
        if loss_ops is None:
            trainable_vars = self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            vars_with_weights = [VariableWithWeight(var) for var in trainable_vars]
            loss_ops = dict(loss=vars_with_weights)

        # Validate operations
        for loss_op_name in loss_ops:
                if loss_op_name not in self._ops:
                    raise ValueError(f'The operation `{loss_op_name}` does not exist.')

        optimizer_ops: List[Tuple[tf.Tensor, tf.Tensor]] = []
        for loss_op_name, vars_with_weights in loss_ops.items():
            # Compute Gradients
            trainable_vars = [var.variable for var in vars_with_weights]
            gradients = tf.gradients(self._ops[loss_op_name], trainable_vars)

            # Clip Gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.hypers.gradient_clip)

            # Prune NoneType values from the set of gradients
            pruned_gradients = [(grad * var.weight, var.variable) for grad, var in zip(clipped_gradients, vars_with_weights) if grad is not None]

            optimizer_op = self._optimizer.apply_gradients(pruned_gradients)
            optimizer_ops.append(optimizer_op)

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

    def train(self, dataset: Dataset, drop_incomplete_batches: bool = False) -> DefaultDict[str, List[float]]:
        """
        Trains the model on the given dataset.

        Args:
            dataset: Dataset object containing training, validation and testing partitions
            drop_incomplete_minibatches: Whether to drop incomplete batches
        Returns:
            A dictionary of metrics obtained from training.
        """
        self.load_metadata(dataset)

        current_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        name = f'{dataset.dataset_name}-{current_date}_model_best'
        if len(self.name) > 0:
            name = f'{self.name}-{name}'

        # Make Model and Initialize variables
        self.make(is_train=True)
        self.init()

        train_loss_dict: DefaultDict[str, List[float]] = defaultdict(list)
        valid_loss_dict: DefaultDict[str, List[float]] = defaultdict(list)
        best_valid_loss_dict: DefaultDict[str, BIG_NUMBER] = defaultdict(lambda: BIG_NUMBER)

        num_not_improved = 0
        for epoch in range(self.hypers.epochs):
            print(f'-------- Epoch {epoch} --------')

            train_generator = dataset.minibatch_generator(DataSeries.TRAIN,
                                                          batch_size=self.hypers.batch_size,
                                                          metadata=self.metadata,
                                                          should_shuffle=True,
                                                          drop_incomplete_batches=drop_incomplete_batches)
            epoch_train_loss: DefaultDict[str, float] = defaultdict(float)
            for i, batch in enumerate(train_generator):
                feed_dict = self.batch_to_feed_dict(batch, is_train=True)
                ops_to_run = [self.optimizer_op_name] + self.loss_op_names
                train_results = self.execute(feed_dict, ops_to_run)

                batch_loss = 0.0
                for loss_op_name in self.loss_op_names:
                    avg_batch_loss = np.average(train_results[loss_op_name])
                    batch_loss += avg_batch_loss
                    train_loss_dict[loss_op_name].append(avg_batch_loss)
                    epoch_train_loss[loss_op_name] += avg_batch_loss

                train_loss_agg = np.average(list(epoch_train_loss.values()))
                avg_train_loss_so_far = train_loss_agg / (i+1)

                print(f'Train Batch {i}. Average loss so far: {avg_train_loss_so_far:.4f}', end='\r')
            print()

            valid_generator = dataset.minibatch_generator(DataSeries.VALID,
                                                          batch_size=self.hypers.batch_size,
                                                          metadata=self.metadata,
                                                          should_shuffle=False,
                                                          drop_incomplete_batches=drop_incomplete_batches)
            epoch_valid_loss: DefaultDict[str, float] = defaultdict(float)
            for i, batch in enumerate(valid_generator):
                feed_dict = self.batch_to_feed_dict(batch, is_train=False)
                valid_results = self.execute(feed_dict, self.loss_op_names)

                batch_loss = 0.0
                for loss_op_name in self.loss_op_names:
                    avg_batch_loss = np.average(valid_results[loss_op_name])
                    batch_loss += avg_batch_loss
                    valid_loss_dict[loss_op_name].append(avg_batch_loss)
                    epoch_valid_loss[loss_op_name] += avg_batch_loss

                valid_loss_agg = np.average(list(epoch_valid_loss.values()))
                avg_valid_loss_so_far = valid_loss_agg / (i+1)

                print(f'Valid Batch {i}. Average loss so far: {avg_valid_loss_so_far:.4f}', end='\r')
            print()

            has_improved = False
            for loss_op_name, valid_loss in epoch_valid_loss.items():
                if valid_loss < best_valid_loss_dict[loss_op_name]:
                    has_improved = True

                    variable_group = self.get_variable_group(loss_op_name)
                    group_dict: Dist[str, List[tf.Variable]] = dict()
                    group_dict[loss_op_name] = variable_group

                    self.save(name=name,
                              variable_groups=group_dict,
                              data_folders=dataset.data_folders)

                    print(f'Saving Model For Operation: {loss_op_name}')

                    best_valid_loss_dict[loss_op_name] = valid_loss

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

        return metrics_dict

    def save(self, name: str, variable_groups: Optional[Dict[str, List[tf.Variable]]] = None,
             data_folders: Optional[Dict[DataSeries, str]] = None):
        """
        Save model weights and hyper-parameters.
        """
        params_path = self.save_folder.join(f'model-hyper-params-{name}.pkl.gz')
        params_path.save_as_compressed_file(self.hypers.__dict__())

        metadata_path = self.save_folder.join(f'model-metadata-{name}.pkl.gz')
        metadata_path.save_as_compressed_file(dict(metadata=self.metadata, data_folders=data_folders))

        with self._sess.graph.as_default():
            if variable_groups is None:
                model_path = self.save_folder.join(f'model-{name}.ckpt')
                saver = tf.train.Saver()
                saver.save(self._sess, model_path.path)
            else:
                # Save variable groups.
                varname_path = self.save_folder.join(f'model-varnames-{name}.pkl.gz')
                variables_dict: Dict[str, List[str]] = dict()
                if varname_path.exists():
                    variables_dict = varname_path.read_by_file_suffix()

                varname_dict = {group_name: [var.name for var in variables] for group_name, variables in variable_groups.items()}
                variables_dict.update(varname_dict)
                varname_path.save_as_compressed_file(variables_dict)

                # Save model weights
                for group_name, var_set in variable_groups.items():
                    model_path = self.save_folder.join(f'model-{name}-{group_name}.ckpt')
                    saver = tf.train.Saver(var_set)
                    saver.save(self._sess, model_path.path)

    def restore_parameters(self, name: str):
        """
        Restore model metadata and hyperparameters.
        """
        params_path = self.save_folder.join(f'model-hyper-params-{name}.pkl.gz')
        self.hypers = extract_hyperparameters(params_path)[0]

        metadata_path = self.save_folder.join(f'model-metadata-{name}.pkl.gz')
        train_metadata = metadata_path.read_by_file_suffix()
        self.metadata = train_metadata['metadata']

    def restore_weights(self, name: str):
        """
        Restore model weights.
        """
        with self._sess.graph.as_default():
            varname_path = self.save_folder.join(f'model-varnames-{name}.pkl.gz')
            varname_dict = varname_path.read_by_file_suffix() if varname_path.exists() else None

            trainable_vars = list(self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

            if varname_dict is not None:
                for loss_op_name, variable_names in varname_dict.items():
                    model_path = self.save_folder.join(f'model-{name}-{loss_op_name}.ckpt')
                    
                    variables = list(filter(lambda v: v.name in variable_names, trainable_vars))
                    saver = tf.train.Saver(variables)
                    saver.restore(self._sess, model_path.path)
            else:
                model_path = self.save_folder.join('model-{name}.ckpt')
                saver = tf.train.Saver()
                saver.restore(self._sess, model_path.path)
