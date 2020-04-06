from utils.hyperparameters import HyperParameters
from .base_model import Model
from .adaptive_model import AdaptiveModel
from .standard_model import StandardModel
from .decision_tree_model import DecisionTreeModel
from .logistic_regression_model import LogisticRegressionModel
from .linear_svm import LinearSVMModel


def get_model(hypers: HyperParameters, save_folder: str, is_train: bool) -> Model:
    model_type = hypers.model.lower()

    if model_type == 'adaptive':
        return AdaptiveModel(hypers, save_folder, is_train)
    elif model_type == 'standard':
        return StandardModel(hypers, save_folder, is_train)
    elif model_type == 'decision_tree':
        return DecisionTreeModel(hypers, save_folder, is_train)
    elif model_type == 'logistic_regression':
        return LogisticRegressionModel(hypers, save_folder, is_train)
    elif model_type == 'linear_svm':
        return LinearSVMModel(hypers, save_folder, is_train)
    else:
        raise ValueError(f'Unknown model type: {model_type}.')
