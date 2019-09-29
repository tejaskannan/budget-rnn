import tensorflow as tf
import numpy as np
from dpu_utils.utils import RichPath
from datetime import datetime
from typing import Optional, Iterable, Dict, Any, Union, List

from dataset.dataset import Dataset, DataSeries
from utils.hyperparameters import HyperParameters
from utils.tfutils import get_optimizer
from utils.file_utils import to_rich_path


class Model:

    def __init__(self, hyper_parameters: HyperParameters, save_folder: Union[str, RichPath]):
        self.hypers = hyper_parameters
        self.save_folder = to_rich_path(save_folder)
        self.metadata: Dict[str, Any] = dict()
        
        self._sess = tf.Session(graph=tf.Graph())
        self._optimizer = get_optimizer(self.hypers.optimizer, self.hypers.learning_rate)
        self._ops: Dict[str, tf.Tensor] = dict()
        self._placeholders: Dict[str, tf.Tensor] = dict()

        # Make the output folder
        self.save_folder.make_as_dir()

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

    def make_training_step(self, trainable_vars: Optional[Iterable[tf.Tensor]] = None):
        """
        Creates the training step for this model. Gradients are clipped
        for better numerical stability.

        Args:
            trainable_vars: Optional set of variables to compute gradients for. If none are provided,
                then the set of all trainable variables is used.
        """
        if 'loss' not in self._ops:
            raise ValueError('Must define a loss before the training step can be created.')

        # Compute Gradients
        if trainable_vars is not None:
            trainable_vars = self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradients = tf.gradients(self._ops['loss'], trainable_vars)

        # Clip Gradients
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.hypers.gradient_clip)

        # Prune NoneType values from the set of gradients
        pruned_gradients = [(grad, var) for grad, var in zip(clipped_gradients, trainable_vars) if grad is not None]

        # Apply the gradients to the specified variables
        self._ops['train_op'] = self._optimizer.apply_gradients(pruned_gradients)

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
            op_results = self._sess.run(ops_to_run, feed_dict)
            return op_results

    def train(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Trains the model on the given dataset.

        Args:
            dataset: Dataset object containing training, validation and testing partitions
        Returns:
            A dictionary of metrics obtained from training.
        """
        self.load_metadata(dataset)

        with self._sess.graph.as_default():
            self.make_placeholders()
            self.make_model()
            self.make_loss()

        current_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        name = '{current_date}_model_best'

        # Initialize variables
        self.init()

        best_valid_loss = 1e10
        for epoch in range(self.hypers.epochs):
            print(f'-------- Epoch {epoch} --------')

            train_losses: List[float] = []
            train_generator = dataset.minibatch_generator(DataSeries.TRAIN,
                                                          batch_size=self.hypers.batch_size,
                                                          metadata=self.metadata,
                                                          should_shuffle=True)
            for i, batch in enumerate(train_generator):
                feed_dict = self.batch_to_feed_dict(batch, is_train=True)
                train_results = self.execute(feed_dict, ['train_op', 'loss'])
                train_losses.append(train_results['loss'])

                avg_loss = np.average(train_losses)
                print(f'Train Batch {i}. Average loss so far: {avg_loss:.2f}', end='\r')
            print()

            valid_losses: List[float] = []
            valid_generator = dataset.minibatch_generator(DataSeries.VALID,
                                                          batch_size=self.hypers.batch_size,
                                                          metadata=self.metadata,
                                                          should_shuffle=False)
            for i, batch in enumerate(valid_generator):
                feed_dict = self.batch_to_feed_dict(batch, is_train=False)
                valid_results = self.execute(feed_dict, 'loss')
                valid_losses.append(valid_results['loss'])

                avg_loss = np.average(valid_losses)
                print(f'Valid Batch {i}. Average loss so far: {avg_loss:.2f}', end='\r')
            print()

            valid_loss = np.average(valid_losses)
            if valid_loss > best_valid_loss:
                num_not_improved += 1
            else:
                self.save(name)
                valid_loss = best_valid_loss
                num_not_improved = 0

            if num_not_improved >= self.hypers.patience:
                print('Exiting due to Early Stopping')
                return


    def save(self, name: str):
        """
        Save model weights and hyper-parameters.
        """
        params_path = self.save_folder.join('hyper_params.pkl.gz')
        params_path.save_as_compressed_file(self.hypers.__dict__())

        metadata_path = self.save_folder.join('metadata.pkl.gz')
        metadata_path.save_as_compressed_file(self.metadata)

        with self._sess.graph.as_default():
            model_path = self.save_folder.join(f'model-{name}.ckpt')
            saver = tf.train.Saver()
            saver.save(self._sess, model_path.path)

    def restore(self, name: str):
        """
        Restore model weights and hyperparameters.
        """
        params_path = self.save_folder.join('hyper_params.pkl.gz')
        self.hypers = HyperParameters(params_path)

        metadata_path = self.save_folder.join('metadata.pkl.gz')
        self.metadata = metadata.read_by_file_suffix()

        with self._sess.graph.as_default():
            model_path = self.save_folder.join(f'model-{name}.ckpt')
            saver = tf.train.Saver()
            saver.restore(self._sess, model_path.path)
