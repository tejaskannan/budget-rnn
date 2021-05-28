import numpy as np
from dataset.dataset import Dataset
from typing import Dict, Any
from datetime import datetime

from utils.constants import DATE_FORMAT, INPUTS, OUTPUT, SAMPLE_ID, INPUT_NOISE, SMALL_NUMBER
from utils.constants import INPUT_SHAPE, INPUT_SCALER, OUTPUT_SCALER, NUM_OUTPUT_FEATURES
from utils.constants import NUM_CLASSES, LABEL_MAP


class SequenceDataset(Dataset):

    def tensorize(self, sample: Dict[str, Any], metadata: Dict[str, Any], is_train: bool) -> Dict[str, np.ndarray]:

        input_shape = metadata[INPUT_SHAPE]
        sequence_length = len(sample[INPUTS])
        inputs = np.array(sample[INPUTS])

        # Normalize inputs
        normalized_input = inputs
        input_scaler = metadata[INPUT_SCALER]
        if input_scaler is not None:
            # Since the standard scaler expects a 2D input, we reshape before normalizing
            input_sample = np.reshape(inputs, newshape=(-1,) + input_shape)
            normalized_input = input_scaler.transform(input_sample)
            normalized_input = np.reshape(normalized_input, newshape=(-1, sequence_length) + input_shape)

        # Add noise to training inputs
        if is_train and metadata.get(INPUT_NOISE, 0.0) > SMALL_NUMBER:
            noise_scale = metadata[INPUT_NOISE]
            input_noise = np.random.uniform(low=-noise_scale, high=noise_scale, size=normalized_input.shape)
            normalized_input = np.array(normalized_input) + input_noise

        # Re-map labels for classification problems
        output = sample[OUTPUT]
        if metadata[NUM_CLASSES] > 0:
            label_map = metadata[LABEL_MAP]

            # Design decision: If the output is not known, then we choose a random
            # label. An unknown label means the label is not present during training.
            # In this situation, we expect the models to perform akin to random guessing.
            if output not in label_map:
                output = np.random.choice(list(label_map.values()))
            else:
                output = label_map[output]

        # Normalize outputs (Scaler expects a 2D input)
        output_scaler = metadata[OUTPUT_SCALER]
        if output_scaler is None:
            normalized_output = [output]
        elif not isinstance(output, list) and not isinstance(output, np.ndarray):
            normalized_output = output_scaler.transform([[output]])
        else:
            normalized_output = output_scaler.transform([output])

        # Shape output into batch
        normalized_output = np.reshape(normalized_output, (-1, metadata[NUM_OUTPUT_FEATURES]))

        batch_dict = {
            INPUTS: normalized_input,
            OUTPUT: normalized_output,
            SAMPLE_ID: sample[SAMPLE_ID]
        }

        return batch_dict
