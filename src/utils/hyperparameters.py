from typing import Union, Dict, Any, List, Optional
from dpu_utils.utils import RichPath
from itertools import product

from utils.file_utils import to_rich_path


class HyperParameters:

    __slots__ = ['epochs', 'patience', 'learning_rate', 'gradient_clip', 'learning_rate_decay', 'optimizer', 'batch_size', 'model', 'dropout_keep_rate', 'model_params']

    def __init__(self, parameters: Dict[str, Any]):
        # Unpack arguments
        self.learning_rate = parameters.get('learning_rate', 0.0001)
        self.gradient_clip = parameters.get('gradient_clip', 1)
        self.learning_rate_decay = parameters.get('learning_rate_decay', 0.99)
        self.dropout_keep_rate = parameters.get('dropout_keep_rate', 1.0)
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
            'dropout_keep_rate': self.dropout_keep_rate,
            'model': self.model,
            'model_params': self.model_params
        }

    def __str__(self) -> str:
        return str(self.__dict__())


def extract_hyperparameters(params_file: Union[str, RichPath],
                            search_fields: Optional[List[str]] = None) -> List[HyperParameters]:
    params_file = to_rich_path(params_file)
    assert params_file.exists(), f'The parameters file {params_file} does not exist!'

    parameters = params_file.read_by_file_suffix()

    if search_fields is None:
        return [HyperParameters(parameters)]

    grid_params: List[List[Any]] = []
    for field in search_fields:
        if field in parameters:
            field_values = parameters[field]
        else:
            field_values = parameters['model_params'][field]

        # Tread non-list grid fields as single-element lists
        if not isinstance(field_values, list):
            grid_params.append([field_values])
        else:
            grid_params.append(field_values)

    grid_permutations = product(*grid_params)

    hyperparameters: List[HyperParameters] = []
    for grid_setting in grid_permutations:
        for field, setting in zip(search_fields, grid_setting):
            if field in parameters:
                parameters[field] = setting
            else:
                parameters['model_params'][field] = setting

        hypers = HyperParameters(parameters)
        hyperparameters.append(hypers)

    return hyperparameters
