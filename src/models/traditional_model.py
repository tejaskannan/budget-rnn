from typing import List, Set, Any, Dict, Optional


from dataset.dataset import Dataset
from utils.constants import INPUTS, OUTPUT, INPUT_SCALER, INPUT_SHAPE, NUM_OUTPUT_FEATURES
from utils.constants import NUM_CLASSES, LABEL_MAP, REV_LABEL_MAP

from .base_model import Model


class TraditionalModel(Model):

    def __init__(self, hyper_parameters: HyperParameters, save_folder: str, is_train: bool):
        super().__init__(hyper_parameters, save_folder, is_train)
        self.name = 'traditional_model'

    def load_metadata(self, dataset: Dataset):
        input_samples: List[np.ndarray] = []
        output_samples: List[Any] = []

        unique_labels: Set[Any] = dict()
        for sample in dataset.iterate_series(series=DataSeries.TRAIN):
            input_sample = np.array(sample[INPUTS]).reshape(-1)
            input_samples.append(input_sample)
            
            output_samples.append(sample[OUTPUT])

            if self.output_type == OutputType.MULTI_CLASSIFICATION:
                unique_labels.add(sample[OUTPUT])

        # Get the number of input features
        num_input_features = len(input_samples[0])

        # Create and fit the input sample scaler
        input_scaler = StandardScaler()
        input_scaler.fit(input_samples)

        # Reshape the output samples into a 1 dimensional array
        output_samples = np.array(output_samples).reshape(-1)

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
        self.metadata[REV_LABEL_MAP] = rev_label_map
