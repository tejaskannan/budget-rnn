import numpy as np
from enum import Enum, auto
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from utils.hyperparameters import HyperParameters
from .traditional_model import TraditionalModel


class DecisionTreeType(Enum):
    STANDARD = auto()
    RANDOM_FOREST = auto()
    ADA_BOOST = auto()


class DecisionTreeModel(TraditionalModel):

    def __init__(self, hyper_parameters: HyperParameters, save_folder: str, is_train: bool):
        super().__init__(hyper_parameters, save_folder, is_train)

        self._tree_type = DecisionTreeType[hyper_parameters.model_params['tree_type'].upper()]
        self._model = None
        self.name = f'decision-tree-{self._tree_type.name}'

    @property
    def tree_type(self) -> DecisionTreeType:
        return self._tree_type

    def make(self, is_train: bool, is_frozen: bool):
        # Prevent making the model multiple times
        if self.model is not None:
            return

        if self.tree_type == DecisionTreeType.STANDARD:
            self._model = DecisionTreeClassifier(criterion=self.hypers.model_params['criterion'],
                                                max_depth=self.hypers.model_params['max_depth'])
        elif self.tree_type == DecisionTreeType.RANDOM_FOREST:
            self._model = RandomForestClassifier(n_estimators=self.hypers.model_params['num_estimators'],
                                                criterion=self.hypers.model_params['criterion'],
                                                max_depth=self.hypers.model_params['max_depth'])
        elif self.tree_type == DecisionTreeType.ADA_BOOST:
            base_estimator = DecisionTreeClassifier(criterion=self.hypers.model_params['criterion'],
                                                    max_depth=self.hypers.model_params['max_depth'])
            self._model = AdaBoostClassifier(base_estimator=base_estimator,
                                             n_estimators=self.hypers.model_params['num_estimators'],
                                             learning_rate=self.hypers.learning_rate)
        else:
            raise ValueError(f'Unknown tree type {self.tree_type}')
