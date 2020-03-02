from dpu_utils.utils import RichPath
from typing import Union

from utils.hyperparameters import HyperParameters
from .rnn_model import RNNModel
from .rnn_sample_model import RNNSampleModel


def get_rnn_model(name: str, hypers: HyperParameters, save_folder: Union[str, RichPath]) -> RNNModel:
    name = name.lower()
    if name in ('rnn_sample', 'sample_model', 'rnn_sample_model'):
        return RNNSampleModel(hypers, save_folder)

    raise ValueError(f'Unknown model name: {name}')
