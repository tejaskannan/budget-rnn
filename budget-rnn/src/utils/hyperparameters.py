import os.path
from typing import Union, Dict, Any, List, Optional
from itertools import product

from utils.file_utils import read_by_file_suffix


class HyperParameters:

    def __init__(self, parameters: Dict[str, Any]):
        # Unpack arguments
        self.learning_rate = parameters.get('learning_rate', 0.0001)
        self.gradient_clip = parameters.get('gradient_clip', 1)
        self.learning_rate_decay = parameters.get('learning_rate_decay', 0.99)
        self.decay_steps = parameters.get('learning_rate_decay_steps', 100000)
        self.dropout_keep_rate = parameters.get('dropout_keep_rate', 1.0)
        self.optimizer = parameters.get('optimizer', 'adam')
        self.batch_size = parameters.get('batch_size', 1000)
        self.epochs = parameters.get('epochs', 10)
        self.patience = parameters.get('patience', 5)
        self.model = parameters.get('model')
        self.model_params = parameters.get('model_params', dict())
        self.input_noise = parameters.get('input_noise', 0.0)
        self.batch_noise = parameters.get('batch_noise', 0.0)
        self.dataset_type = parameters.get('dataset_type', 'standard')
        self.seq_length = parameters.get('seq_length')

    def as_dict(self) -> Dict[str, Any]:
        return {
            'epochs': self.epochs,
            'patience': self.patience,
            'learning_rate': self.learning_rate,
            'gradient_clip': self.gradient_clip,
            'learning_rate_decay': self.learning_rate_decay,
            'decay_steps': self.decay_steps,
            'optimizer': self.optimizer,
            'batch_size': self.batch_size,
            'dropout_keep_rate': self.dropout_keep_rate,
            'model': self.model,
            'model_params': self.model_params,
            'input_noise': self.input_noise,
            'batch_noise': self.batch_noise,
            'dataset_type': self.dataset_type,
            'seq_length': self.seq_length
        }

    def __str__(self) -> str:
        return str(self.as_dict())

    @classmethod
    def create_from_file(cls, params_file: str):
        """
        Reads the given hyper parameters from the serialized file.
        """
        assert os.path.exists(params_file), f'The parameters file {params_file} does not exist!'

        parameters = read_by_file_suffix(params_file)
        return HyperParameters(parameters)
