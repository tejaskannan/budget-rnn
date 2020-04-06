from sklearn.svm import LinearSVC

from utils.hyperparameters import HyperParameters
from .traditional_model import TraditionalModel


class LinearSVMModel(TraditionalModel):

    def __init__(self, hyper_parameters: HyperParameters, save_folder: str, is_train: bool):
        super().__init__(hyper_parameters, save_folder, is_train)

        self._model = None
        self.name = 'linear-svm'

    def make(self, is_train: bool, is_frozen: bool):
        if self.model is not None:
            return

        self._model = LinearSVC(penalty=self.hypers.model_params['penalty'],
                                C=self.hypers.model_params['regularization_strength'],
                                dual=False,
                                max_iter=self.hypers.model_params['max_iters'])
