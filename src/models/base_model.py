import numpy as np
import re
import os
from collections import defaultdict
from typing import Optional, Iterable, Dict, Any, Union, List, DefaultDict, Set

from dataset.dataset import Dataset, DataSeries
from layers.output_layers import OutputType
from utils.hyperparameters import HyperParameters
from utils.file_utils import make_dir


class Model:

    def __init__(self, hyper_parameters: HyperParameters, save_folder: str, is_train: bool):
        self.hypers = hyper_parameters
        self.save_folder = save_folder
        self.metadata: Dict[str, Any] = dict()

        # Get the model output type
        self._output_type = OutputType[self.hypers.model_params['output_type'].upper()]

        make_dir(self.save_folder)
        self.name = 'model'  # Default name

    @property
    def output_type(self) -> OutputType:
        return self._output_type

    def load_metadata(self, dataset: Dataset):
        """
        Loads metadata from the dataset. Results are stored
        directly into self.metadata.
        """
        pass

    def predict(self, dataset: Dataset,
                test_batch_size: Optional[int],
                max_num_batches: Optional[int],
                series: DataSeries = DataSeries.TEST) -> DefaultDict[str, Dict[str, Any]]:
        """
        Execute the model to produce a prediction for the given input sample.

        Args:
            dataset: Dataset object used to create input tensors
            test_batch_size: Batch size to use during testing
            max_num_batches: Maximum number of batches to perform testing on
            series: Data series from which to compute predictions for
        Returns:
            The predicted output produced by the model.
        """
        test_batch_size = test_batch_size if test_batch_size is not None else self.hypers.batch_size
        test_batch_generator = dataset.minibatch_generator(series=series,
                                                           batch_size=test_batch_size,
                                                           metadata=self.metadata,
                                                           should_shuffle=False)

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

    def init(self):
        """
        Initializes all variables in the computation graph.
        """
        pass

    def make(self, is_train: bool, is_frozen: bool):
        """
        Creates model and optimizer op.

        Args:
            is_train: Whether the model is built for training or just for inference.
            is_frozen: Whether the mode ls built with frozen inputs.
        """
        pass

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
        pass

    def save(self, name: str, data_folders: Dict[DataSeries, str], loss_ops: Optional[List[str]], loss_var_dict: Dict[str, List[str]]):
        """
        Save model weights, hyper-parameters, and metadata

        Args:
            name: Name of the model
            data_folders: Data folders used for training and validation
            loss_ops: Loss operations for which to save variables. None value indicates that ALL variables
                are to be saved
        """
        pass

    def restore(self, name: str, is_train: bool, is_frozen: bool):
        """
        Restore model metadata, hyper-parameters, and trainable parameters.
        """
        pass
