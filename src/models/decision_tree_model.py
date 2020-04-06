import numpy as np
import os
import time
from datetime import datetime
from enum import Enum, auto
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from typing import Optional, Dict, List, Any, Iterable, DefaultDict

from layers.output_layers import OutputType
from dataset.dataset import Dataset, DataSeries
from utils.constants import INPUTS, OUTPUT, NAME_FMT, TRAIN_LOG_PATH, INPUT_SHAPE, PREDICTION
from utils.constants import MODEL_PATH, HYPERS_PATH, METADATA_PATH, MODEL, NUM_CLASSES
from utils.file_utils import save_by_file_suffix, read_by_file_suffix
from utils.hyperparameters import HyperParameters
from utils.testing_utils import get_binary_classification_metric, get_multi_classification_metric, ALL_LATENCY, ClassificationMetric
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
        self.name = f'decision-tree-{self._tree_type.name}'

    @property
    def tree_type(self) -> DecisionTreeType:
        return self._tree_type

    @property
    def tree(self) -> Optional[DecisionTreeClassifier]:
        return self._tree

    def make(self, is_train: bool, is_frozen: bool):
        # Prevent making the model multiple times
        if self._tree is not None:
            return

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
                                            learning_rate=self.hypers.learning_rate)
        else:
            raise ValueError(f'Unknown tree type {self.tree_type}')
                                                
    def train(self, dataset: Dataset, drop_incomplete_batches: bool = False) -> str:
        """
        Fits the decision tree model
        """
        # Load the metadata
        self.load_metadata(dataset)

        current_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        name = NAME_FMT.format(self.name, dataset.dataset_name, current_date)

        # Create the model
        self.make(is_train=True, is_frozen=False)

        # Fit the decision tree on the training set
        train_inputs: List[np.ndarray] = []
        train_outputs: List[int] = []
        train_generator = dataset.minibatch_generator(DataSeries.TRAIN,
                                                      batch_size=self.hypers.batch_size,
                                                      metadata=self.metadata,
                                                      should_shuffle=False,
                                                      drop_incomplete_batches=drop_incomplete_batches)
        for batch in train_generator:
            train_inputs.extend(batch[INPUTS])
            train_outputs.extend(batch[OUTPUT])

        train_inputs = np.reshape(train_inputs, newshape=(-1, self.metadata[INPUT_SHAPE]))
        train_outputs = np.reshape(train_outputs, newshape=(-1))

        print('Fitting the model...')
        self.tree.fit(train_inputs, train_outputs)
        print('Completed training. Starting validation...')

        # Validate model on the validation set
        valid_inputs: List[np.ndarray] = []
        valid_outputs: List[int] = []
        valid_generator = dataset.minibatch_generator(DataSeries.VALID,
                                                      batch_size=self.hypers.batch_size,
                                                      metadata=self.metadata,
                                                      should_shuffle=False,
                                                      drop_incomplete_batches=drop_incomplete_batches)
        for batch in valid_generator:
            valid_inputs.extend(batch[INPUTS])
            valid_outputs.extend(batch[OUTPUT])

        valid_inputs = np.reshape(valid_inputs, newshape=(-1, self.metadata[INPUT_SHAPE]))
        valid_outputs = np.reshape(valid_outputs, newshape=(-1))
        valid_accuracy = self.tree.score(valid_inputs, valid_outputs)

        print(f'Validation Accuracy: {valid_accuracy:.4f}')

        # Save results
        train_results = dict(valid_accuracy=valid_accuracy)
        log_file = os.path.join(self.save_folder, TRAIN_LOG_PATH.format(name))
        save_by_file_suffix(train_results, log_file)

        self.save(name, dataset.data_folders, None, None)

        return name

    def predict_classification(self, test_batch_generator: Iterable[Dict[str, Any]],
                               test_batch_size: int,
                               max_num_batches: Optional[int],
                               flops_dict: Dict[str, Any]) -> DefaultDict[str, Dict[str, Any]]:
        predictions_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []
        latencies: List[float] = []

        for batch_num, test_batch in enumerate(test_batch_generator):

            inputs = np.reshape(test_batch[INPUTS], newshape=(-1, self.metadata[INPUT_SHAPE]))

            start = time.time()
            predictions = self.tree.predict(inputs)
            elapsed = time.time() - start

            labels_list.append(np.vstack(test_batch[OUTPUT]))
            predictions_list.append(np.vstack(predictions))
            latencies.append(elapsed)

            if max_num_batches is not None and batch_num >= max_num_batches:
                break

        labels = np.vstack(labels_list)
        predictions = np.vstack(predictions_list)
        avg_latency = np.average(latencies[1:])
        flops = flops_dict[PREDICTION]

        result: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
        for metric_name in ClassificationMetric:
            if self.output_type == OutputType.BINARY_CLASSIFICATION:
                metric_value = get_binary_classification_metric(metric_name, predictions, labels, avg_latency, 1, flops)
            else:
                metric_value = get_multi_classification_metric(metric_name, predictions, labels, avg_latency, 1, flops, self.metadata[NUM_CLASSES])

            result[MODEL][metric_name.name] = metric_value

        result[MODEL][ALL_LATENCY] = latencies[1:]

        return result

    def save(self, name: str, data_folders: Dict[DataSeries, str], loss_ops: Optional[List[str]], loss_var_dict: Optional[Dict[str, List[str]]]):
        """
        Serializes the parameters of the decision tree.
        """
        # Save hyperparameters
        params_path = os.path.join(self.save_folder, HYPERS_PATH.format(name))
        save_by_file_suffix(self.hypers.__dict__(), params_path)

        # Save metadata
        data_folders_dict = {series.name: path for series, path in data_folders.items()}
        metadata_path = os.path.join(self.save_folder, METADATA_PATH.format(name))
        save_by_file_suffix(dict(metadata=self.metadata, data_folders=data_folders_dict), metadata_path)

        # Save model parameters
        model_path = os.path.join(self.save_folder, MODEL_PATH.format(name))
        save_by_file_suffix(self._tree, model_path)

    def restore(self, name: str, is_train: bool, is_frozen: bool):
        # Restore hyperparameters
        params_path = os.path.join(self.save_folder, HYPERS_PATH.format(name))
        self.hypers = HyperParameters.create_from_file(params_path)

        # Restore metadata
        metadata_path = os.path.join(self.save_folder, METADATA_PATH.format(name))
        train_metadata = read_by_file_suffix(metadata_path)
        self.metadata = train_metadata['metadata']

        # Build the decision tree
        self.make(is_train=is_train, is_frozen=is_frozen)

        # Restore the learned parameters
        model_path = os.path.join(self.save_folder, MODEL_PATH.format(name))
        self._tree = read_by_file_suffix(model_path)
