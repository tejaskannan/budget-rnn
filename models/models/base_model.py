import tensorflow as tf
import numpy as np
from dpu_utils.utils import RichPath
from datetime import datetime
from collections import defaultdict
from typing import Optional, Iterable, Dict, Any, Union, List, DefaultDict

from dataset.dataset import Dataset, DataSeries
from utils.hyperparameters import HyperParameters
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

        self._ops['optimizer_op'] = tf.group(optimizer_ops)

        # Apply the gradients to the specified variables
        # self._ops['optimizer_op'] = self._optimizer.apply_gradients(pruned_gradients)

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

    def train(self, dataset: Dataset) -> DefaultDict[str, List[float]]:
        """
        Trains the model on the given dataset.

        Args:
            dataset: Dataset object containing training, validation and testing partitions
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

        metrics_dict: DefaultDict[str, List[float]] = defaultdict(list)

        best_valid_loss = BIG_NUMBER
        num_not_improved = 0
        for epoch in range(self.hypers.epochs):
            print(f'-------- Epoch {epoch} --------')

            train_losses: List[float] = []
            train_generator = dataset.minibatch_generator(DataSeries.TRAIN,
                                                          batch_size=self.hypers.batch_size,
                                                          metadata=self.metadata,
                                                          should_shuffle=True,
                                                          drop_incomplete_batches=True)

            for i, batch in enumerate(train_generator):
                feed_dict = self.batch_to_feed_dict(batch, is_train=True)
                train_results = self.execute(feed_dict, ['optimizer_op', 'loss'])
                
                train_losses.append(train_results['loss'])

                avg_loss = np.average(train_losses)
                print(f'Train Batch {i}. Average loss so far: {avg_loss:.5f}', end='\r')
            print()

            valid_losses: List[float] = []
            valid_generator = dataset.minibatch_generator(DataSeries.VALID,
                                                          batch_size=self.hypers.batch_size,
                                                          metadata=self.metadata,
                                                          should_shuffle=False,
                                                          drop_incomplete_batches=True)
            for i, batch in enumerate(valid_generator):
                feed_dict = self.batch_to_feed_dict(batch, is_train=False)
                valid_results = self.execute(feed_dict, ['loss'])
                valid_losses.append(valid_results['loss'])

                avg_loss = np.average(valid_losses)
                print(f'Valid Batch {i}. Average loss so far: {avg_loss:5f}', end='\r')
            print()

            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)

            metrics_dict['train'].append(train_loss)
            metrics_dict['valid'].append(valid_loss)

            if valid_loss >= best_valid_loss:
                num_not_improved += 1
            else:
                print('Saving model...')
                self.save(name, dataset.data_folders)
                best_valid_loss = valid_loss
                num_not_improved = 0

            if num_not_improved >= self.hypers.patience:
                print('Exiting due to Early Stopping')
                break

        return metrics_dict

    def save(self, name: str, data_folders: Optional[Dict[DataSeries, str]] = None):
        """
        Save model weights and hyper-parameters.
        """
        params_path = self.save_folder.join(f'model-hyper-params-{name}.pkl.gz')
        params_path.save_as_compressed_file(self.hypers.__dict__())

        metadata_path = self.save_folder.join(f'model-metadata-{name}.pkl.gz')
        metadata_path.save_as_compressed_file(dict(metadata=self.metadata, data_folders=data_folders))
    
        with self._sess.graph.as_default():
            model_path = self.save_folder.join(f'model-{name}.ckpt')
            saver = tf.train.Saver()
            saver.save(self._sess, model_path.path)

    def restore_parameters(self, name: str):
        """
        Restore model metadata and hyperparameters.
        """
        params_path = self.save_folder.join(f'model-hyper-params-{name}.pkl.gz')
        self.hypers = HyperParameters(params_path)

        metadata_path = self.save_folder.join(f'model-metadata-{name}.pkl.gz')
        train_metadata = metadata_path.read_by_file_suffix()
        self.metadata = train_metadata['metadata']

    def restore_weights(self, name: str):
        """
        Restore model weights.
        """
        with self._sess.graph.as_default():
            model_path = self.save_folder.join(f'model-{name}.ckpt')
            saver = tf.train.Saver()
            saver.restore(self._sess, model_path.path)
