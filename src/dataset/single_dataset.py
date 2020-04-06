import numpy as np
from typing import Dict, Any

from dataset.dataset import Dataset
from utils.constants import INPUT_SHAPE, INPUTS, OUTPUT, SAMPLE_ID, INPUT_NOISE, SMALL_NUMBER
from utils.constants import INPUT_SCALER, NUM_OUTPUT_FEATURES, NUM_CLASSES, LABEL_MAP


class SingleDataset(Dataset):

    def tensorize(self, sample: Dict[str, Any], metadata: Dict[str, Any], is_train: bool) -> Dict[str, np.ndarray]:

        # Normalize inputs
        input_shape = metadata[INPUT_SHAPE]
        input_sample = np.array(sample[INPUTS]).reshape((-1, input_shape))
        input_scaler = metadata[INPUT_SCALER]
        normalized_input = input_scaler.transform(input_sample)  # [1, L * D]

        # Apply input noise during training
        if is_train and metadata.get(INPUT_NOISE, 0.0) > SMALL_NUMBER:
            input_noise = np.random.normal(loc=0.0, scale=metadata[INPUT_NOISE], size=normalized_input.shape)
            normalized_input += input_noise

        # Re-map labels for classification problems
        output = sample[OUTPUT]
        if metadata[NUM_CLASSES] > 0:
            label_map = metadata[LABEL_MAP]
            output = label_map[output]

        return {
            INPUTS: normalized_input,
            OUTPUT: output,
            SAMPLE_ID: sample[SAMPLE_ID]
        }
