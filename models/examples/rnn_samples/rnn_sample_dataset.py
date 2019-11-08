import numpy as np
from dataset.dataset import Dataset
from typing import Dict, Any


class RNNSampleDataset(Dataset):

    def tensorize(self, sample: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, np.ndarray]:

        num_input_features = metadata['num_input_features']
        sequence_length = len(sample['inputs'])
        
        inputs = np.array(sample['inputs'])        
        if metadata.get('shift_inputs', False):
            first_input = np.expand_dims(inputs[0, :], axis=0)
            shifted_input = inputs - first_input
            input_sample = np.reshape(shifted_input, newshape=(-1, num_input_features))
        else:
            input_sample = np.reshape(inputs, newshape=(-1, num_input_features))

        # Normalize inputs
        normalized_input = metadata['input_scaler'].transform(input_sample)
        
        # Normalize outputs (Scaler expects a 2D input)
        if not isinstance(sample['output'], list) and not isinstance(sample['output'], np.ndarray):
            normalized_output = metadata['output_scaler'].transform([[sample['output']]])
        else:
            normalized_output = metadata['output_scaler'].transform([sample['output']])
        
        # Shape into batches
        normalized_output = np.reshape(normalized_output, (-1, metadata['num_output_features']))
        normalized_input = np.reshape(normalized_input, newshape=(-1, sequence_length, num_input_features))

        batch_dict = {
            'inputs': normalized_input,
            'output': normalized_output,
        }

        if 'bin_means' in metadata:
            batch_dict['bin_means'] = np.array(metadata['bin_means'])

        return batch_dict
