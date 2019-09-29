import tensorflow as tf
from dpu_utils.utils import RichPath
from typing import Optional, Iterable, Dict, Any, Union, List

from dataset.dataset import Dataset
from utils.hyperparameters import HyperParameters
from utils.tfutils import get_optimizer
from utils.file_utils import to_rich_path


class Model:

    def __init__(self, hyper_parameters: HyperParameters, save_folder: Union[str, RichPath]):
        self.hypers = hyper_parameters
        self.save_folder = to_rich_path(save_folder)
        self.metadata: Dict[str, Any] = dict()
        
        self.__sess = tf.Session(graph=tf.Graph())
        self.__optimizer = get_optimizer(self.hypers.optimizer)
        self.__ops: Dict[str, tf.Tensor] = dict()
        self.__placeholders: Dict[str, tf.Tensor] = dict()

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

    def init(self):
        """
        Initializes all variables in the computation graph.
        """
        with self.__sess.graph.as_default():
            init_op = tf.global_variables_initializer()
            self.__sess.run(init_op)

    def make_training_step(self, trainable_vars: Optional[Iterable[tf.Tensor]] = None):
        """
        Creates the training step for this model. Gradients are clipped
        for better numerical stability.

        Args:
            trainable_vars: Optional set of variables to compute gradients for. If none are provided,
                then the set of all trainable variables is used.
        """
        if 'loss' not in self.__ops:
            raise ValueError('Must define a loss before the training step can be created.')

        # Compute Gradients
        if trainable_vars is not None:
            trainable_vars = self.__sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradients = tf.gradients(self.__ops['loss'], trainable_vars)

        # Clip Gradients
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.hypers.gradient_clip)

        # Prune NoneType values from the set of gradients
        pruned_gradients = [(grad, var) for grad, var in zip(clipped_gradients, trainable_vars) if grad is not None]

        # Apply the gradients to the specified variables
        self.__optimizer.apply_gradients(pruned_gradients)

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
        ops_to_run = self.ops
        if ops is not None:
            ops_to_run = {op_name: op_val for op_name, op_val in self.ops.items() if op_name in ops}

        with self.__sess.graph.as_default():
            op_results = self.__sess.run(ops_to_run, feed_dict)
            return op_results

    def save(self, name: str):
        """
        Save model weights and hyper-parameters.
        """
        params_path = self.save_folder.join('hyper_params.pkl.gz')
        params_path.save_as_compressed_file(self.hypers.__dict__())

        metadata_path = self.save_folder.join('metadata.pkl.gz')
        metadata_path.save_as_compressed_file(self.metadata)

        with self.__sess.graph.as_default():
            model_path = self.save_folder.join(f'model-{name}.ckpt')
            saver = tf.train.Saver()
            saver.save(self.__sess, model_path.path)

    def restore(self, name: str):
        """
        Restore model weights and hyperparameters.
        """
        params_path = self.save_folder.join('hyper_params.pkl.gz')
        self.hypers = HyperParameters(params_path)

        metadata_path = self.save_folder.join('metadata.pkl.gz')
        self.metadata = metadata.read_by_file_suffix()

        with self.__sess.graph.as_default():
            model_path = self.save_folder.join(f'model-{name}.ckpt')
            saver = tf.train.Saver()
            saver.restore(self.__sess, model_path.path)
