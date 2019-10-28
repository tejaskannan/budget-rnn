import numpy as np
from dataset.dataset import Dataset
from typing import Dict, Any


class RNNSampleDataset(Dataset):

    def tensorize(self, sample: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, np.ndarray]:

        num_input_features = metadata['num_input_features']
        sequence_length = len(sample['inputs'])
        input_sample = np.reshape(sample['inputs'], newshape=(-1, num_input_features))

        # Normalize inputs
        normalized_input = metadata['input_scaler'].transform(input_sample)
        
        # Normalize outputs (Scaler expects a 2D input)
        if not isinstance(sample['output'], list) and not isinstance(sample['output'], np.ndarray):
            normalized_output = metadata['output_scaler'].transform([[sample['output']]])
        else:
            normalized_output = metadata['output_scaler'].transform([sample['output']])

        normalized_input = np.reshape(normalized_input, newshape=(-1, sequence_length, num_input_features))
        return {
            'inputs': np.array(normalized_input),
            'output': np.array(normalized_output).reshape(-1, metadata['num_output_features'])
        }
