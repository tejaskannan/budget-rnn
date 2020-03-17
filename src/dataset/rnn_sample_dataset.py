import numpy as np
from dataset.dataset import Dataset
from typing import Dict, Any
from datetime import datetime

from utils.constants import DATE_FORMAT


class RNNSampleDataset(Dataset):

    def tensorize(self, sample: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, np.ndarray]:

        input_shape = metadata['input_shape']
        sequence_length = len(sample['inputs'])
        inputs = np.array(sample['inputs'])
        
        # Normalize inputs
        normalized_input = inputs
        input_scaler = metadata['input_scaler']
        if input_scaler is not None:
            # Since the standard scaler expects a 2D input, we reshape before normalizing
            input_sample = np.reshape(inputs, newshape=(-1,) + input_shape)
            normalized_input = input_scaler.transform(input_sample)
            normalized_input = np.reshape(normalized_input, newshape=(-1, sequence_length) + input_shape)

        # Normalize outputs (Scaler expects a 2D input)
        output_scaler = metadata['output_scaler']
        if output_scaler is None:
            normalized_output = [sample['output']]
        elif not isinstance(sample['output'], list) and not isinstance(sample['output'], np.ndarray):
            normalized_output = output_scaler.transform([[sample['output']]])
        else:
            normalized_output = output_scaler.transform([sample['output']])

        # Shape output into batch
        normalized_output = np.reshape(normalized_output, (-1, metadata['num_output_features']))

        # Retrieve the sample id
        sample_id = sample['sample_id']

        batch_dict = {
            'inputs': normalized_input,
            'output': normalized_output,
            'sample_id': sample_id
        }

        return batch_dict
