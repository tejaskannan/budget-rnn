import numpy as np
from dataset.dataset import Dataset
from typing import Dict, Any
from datetime import datetime

from utils.constants import DATE_FORMAT, INPUTS, OUTPUT, SAMPLE_ID
from utils.constants import INPUT_SHAPE, INPUT_SCALER, OUTPUT_SCALER, NUM_OUTPUT_FEATURES


class RNNSampleDataset(Dataset):

    def tensorize(self, sample: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, np.ndarray]:

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

        # Normalize outputs (Scaler expects a 2D input)
        output_scaler = metadata[OUTPUT_SCALER]
        if output_scaler is None:
            normalized_output = [sample[OUTPUT]]
        elif not isinstance(sample[OUTPUT], list) and not isinstance(sample[OUTPUT], np.ndarray):
            normalized_output = output_scaler.transform([[sample[OUTPUT]]])
        else:
            normalized_output = output_scaler.transform([sample[OUTPUT]])

        # Shape output into batch
        normalized_output = np.reshape(normalized_output, (-1, metadata[NUM_OUTPUT_FEATURES]))

        batch_dict = {
            INPUTS: normalized_input,
            OUTPUT: normalized_output,
            SAMPLE_ID: sample[SAMPLE_ID]
        }

        return batch_dict
