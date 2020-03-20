from utils.hyperparameters import HyperParameters
from .base_model import Model
from .adaptive_model import AdaptiveModel
from .standard_model import StandardModel


def get_model(hypers: HyperParameters, save_folder: str) -> Model:
    model_type = hypers.model.lower()

    if model_type == 'adaptive':
        return AdaptiveModel(hypers, save_folder)
    elif model_type == 'standard':
        return StandardModel(hypers, save_folder)
    else:
        raise ValueError(f'Unknown model type: {model_type}.')
