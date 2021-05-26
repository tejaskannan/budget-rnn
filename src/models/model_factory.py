from utils.hyperparameters import HyperParameters
from .base_model import Model
from .adaptive_model import AdaptiveModel
from .standard_model import StandardModel


def get_model(hypers: HyperParameters, save_folder: str, is_train: bool) -> Model:
    model_type = hypers.model.lower()

    if model_type == 'adaptive':
        return AdaptiveModel(hypers, save_folder, is_train)
    elif model_type == 'standard':
        return StandardModel(hypers, save_folder, is_train)
    else:
        raise ValueError('Unknown model type: {0}.'.format(model_type))
