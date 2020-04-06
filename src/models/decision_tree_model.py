from enum import Enum, auto
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensembler import AdaBoostClassifier, RandomForestClassifier
from typing import Optional, Dict, List, Any

from .traditional_model import TraditionalModel


class DecisionTreeType(Enum):
    STANDARD = auto()
    RANDOM_FOREST = auto()
    ADA_BOOST = auto()


class DecisionTreeModel(TraditionalModel):

    def __init__(self, hyper_parameters: HyperParameters, save_folder: str, is_train: bool):
        super().__init__(hyper_parameters, save_folder, is_train)

        self._tree_type = DecisionTreeType[hyper_parameters.model_params['tree_type'].upper()]
        self._tree = None

    @property
    def tree_type(self) -> DecisionTreeType:
        return self._tree_type

    @property
    def tree(self) -> Optional[DecisionTreeClassifier]
        return self._tree


    def make(self, is_train: bool, is_frozen: bool):
        if self.tree_type == DecisionTreeType.STANDARD:
            self._tree = DecisionTreeClassifier(criterion=self.hypers.model_params['criterion'],
                                                max_depth=self.hypers.model_params['max_depth'])
        elif self.tree_type == DecisionTreeType.RANDOM_FOREST:
            self._tree = RandomForestClassifier(n_estimators=self.hypers.model_params['num_estimators'],
                                                criterion=self.hypers.model_params['criterion'],
                                                max_depth=self.hypers.model_params['max_depth'])
        elif self.tree_type == DecisionTreeType.ADA_BOOST:
            base_estimator = DecisionTreeClassifier(criterion=self.hypers.model_params['criterion'],
                                                    max_depth=self.hypers.model_params['max_depth'])
            self._tree = AdaBoostClassifier(base_estimator=base_estimator,
                                            learning_rate=self.hypers.model_params['learning_rate'])
        else:
            raise ValueError(f'Unknown tree type {self.tree_type}')
                                                
    def train(self, dataset: Dataset, drop_incomplete_batches: bool = False) -> str:
        """
        Fits the decision tree model
        """
        pass

