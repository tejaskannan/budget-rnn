from typing import Union, Dict, Any
from dpu_utils.utils import RichPath


class HyperParameters:

    __slots__ = ['epochs', 'patience', 'learning_rate', 'gradient_clip', 'learning_rate_decay', 'optimizer', 'batch_size', 'model', 'model_params']

    def __init__(self, params_file: Union[str, RichPath]):
        # Read the parameters file
        if isinstance(params_file, str):
            params_file = RichPath.create(params_file)

        if not params_file.exists():
            raise ValueError(f'The parameters file {params_file} does not exist!')

        parameters = params_file.read_by_file_suffix()

        # Unpack arguments
        self.learning_rate = parameters.get('learning_rate', 0.01)
        self.gradient_clip = parameters.get('gradient_clip', 1)
        self.learning_rate_decay = parameters.get('learning_rate_decay', 0.99)
        self.optimizer = parameters.get('optimizer', 'adam')
        self.batch_size = parameters.get('batch_size', 1000)
        self.epochs = parameters.get('epochs', 10)
        self.patience = parameters.get('patience', 5)
        self.model = parameters.get('model')
        self.model_params = parameters.get('model_params', dict())

    def __dict__(self) -> Dict[str, Any]:
        return {
            'epochs': self.epochs,
            'patience': self.patience,
            'learning_rate': self.learning_rate,
            'gradient_clip': self.gradient_clip,
            'learning_rate_decay': self.learning_rate_decay,
            'optimizer': self.optimizer,
            'batch_size': self.batch_size,
            'model': self.model,
            'model_params': self.model_params
        }

    def __str__(self) -> str:
        return str(self.__dict__())
