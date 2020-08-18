import os
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from typing import List, Set, Any, Dict, Optional, Iterable, DefaultDict

from layers.output_layers import OutputType
from dataset.dataset import Dataset, DataSeries
from utils.constants import INPUTS, OUTPUT, INPUT_SCALER, INPUT_SHAPE, NUM_OUTPUT_FEATURES
from utils.constants import NUM_CLASSES, LABEL_MAP, REV_LABEL_MAP, PREDICTION, TRAIN_LOG_PATH
from utils.constants import MODEL_PATH, HYPERS_PATH, METADATA_PATH, NAME_FMT, MODEL
from utils.file_utils import save_by_file_suffix, read_by_file_suffix
from utils.hyperparameters import HyperParameters
from utils.testing_utils import get_binary_classification_metric, get_multi_classification_metric, ALL_LATENCY, ClassificationMetric

from .base_model import Model


class TraditionalModel(Model):

    def __init__(self, hyper_parameters: HyperParameters, save_folder: str, is_train: bool):
        super().__init__(hyper_parameters, save_folder, is_train)
        self.name = 'traditional_model'
        self._model = None

    @property
    def output_ops(self) -> List[str]:
        return [PREDICTION]

    @property
    def model(self) -> Any:
        return self._model

    def compute_flops(self, level: int) -> int:
        return 0

    def load_metadata(self, dataset: Dataset):
        input_samples: List[np.ndarray] = []
        output_samples: List[Any] = []

        unique_labels: Set[Any] = set()
        for sample in dataset.iterate_series(series=DataSeries.TRAIN):
            input_sample = np.array(sample[INPUTS]).reshape(-1)
            if np.any(np.isnan(input_sample)) or np.any(input_sample is None):
                continue

            if sample[OUTPUT] is None:
                continue

            input_samples.append(input_sample)

            if self.output_type == OutputType.MULTI_CLASSIFICATION:
                unique_labels.add(sample[OUTPUT])

        # Get the number of input features
        num_input_features = len(input_samples[0])

        # Create and fit the input sample scaler
        input_scaler = StandardScaler()
        input_scaler.fit(input_samples)

        # Make the label maps for classification problems
        label_map: Dict[Any, int] = dict()
        reverse_label_map: Dict[int, Any] = dict()
        if self.output_type == OutputType.MULTI_CLASSIFICATION:
            for index, label in enumerate(sorted(unique_labels)):
                label_map[label] = index
                reverse_label_map[index] = label

        # Save values into the metadata dictionary
        self.metadata[INPUT_SCALER] = input_scaler
        self.metadata[INPUT_SHAPE] = num_input_features
        self.metadata[NUM_OUTPUT_FEATURES] = 1  # Only supports scalar outputs
        self.metadata[NUM_CLASSES] = len(label_map)
        self.metadata[LABEL_MAP] = label_map
        self.metadata[REV_LABEL_MAP] = reverse_label_map

    def train(self, dataset: Dataset, drop_incomplete_batches: bool = False) -> str:
        """
        Fits the model to the dataset
        """
        # Load the metadata
        self.load_metadata(dataset)

        current_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        name = NAME_FMT.format(self.name, dataset.dataset_name, current_date)

        # Create the model
        self.make(is_train=True, is_frozen=False)

        # Fit the model on the training set
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
        self.model.fit(train_inputs, train_outputs)
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
        valid_accuracy = self.model.score(valid_inputs, valid_outputs)

        print(f'Validation Accuracy: {valid_accuracy:.4f}')

        # Save results
        train_results = dict(valid_accuracy=valid_accuracy)
        log_file = os.path.join(self.save_folder, TRAIN_LOG_PATH.format(name))
        save_by_file_suffix(train_results, log_file)

        self.save(name, dataset.data_folders, None, None)

        return name

    def predict_classification(self, test_batch_generator: Iterable[Dict[str, Any]],
                               test_batch_size: int,
                               max_num_batches: Optional[int]) -> DefaultDict[str, Dict[str, Any]]:
        predictions_list: List[np.ndarray] = []
        labels_list: List[np.ndarray] = []

        for batch_num, test_batch in enumerate(test_batch_generator):

            inputs = np.reshape(test_batch[INPUTS], newshape=(-1, self.metadata[INPUT_SHAPE]))
            predictions = self.model.predict(inputs)

            labels_list.append(np.vstack(test_batch[OUTPUT]))
            predictions_list.append(np.vstack(predictions))

            if max_num_batches is not None and batch_num >= max_num_batches:
                break

        labels = np.vstack(labels_list)
        predictions = np.vstack(predictions_list)

        result: DefaultDict[str, Dict[str, float]] = defaultdict(dict)
        for metric_name in ClassificationMetric:
            if self.output_type == OutputType.BINARY_CLASSIFICATION:
                metric_value = get_binary_classification_metric(metric_name, predictions, labels)
            else:
                metric_value = get_multi_classification_metric(metric_name, predictions, labels, self.metadata[NUM_CLASSES])

            result[MODEL][metric_name.name] = metric_value

        return result

    def save(self, name: str, data_folders: Dict[DataSeries, str], loss_ops: Optional[List[str]], loss_var_dict: Optional[Dict[str, List[str]]]):
        """
        Serializes the parameters of the model.
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
        save_by_file_suffix(self._model, model_path)

    def restore(self, name: str, is_train: bool, is_frozen: bool):
        # Restore hyperparameters
        params_path = os.path.join(self.save_folder, HYPERS_PATH.format(name))
        self.hypers = HyperParameters.create_from_file(params_path)

        # Restore metadata
        metadata_path = os.path.join(self.save_folder, METADATA_PATH.format(name))
        train_metadata = read_by_file_suffix(metadata_path)
        self.metadata = train_metadata['metadata']

        # Build the model
        self.make(is_train=is_train, is_frozen=is_frozen)

        # Restore the learned parameters
        model_path = os.path.join(self.save_folder, MODEL_PATH.format(name))
        self._model = read_by_file_suffix(model_path)
