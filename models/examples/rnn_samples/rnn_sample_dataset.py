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
        
        # Shape into batches
        normalized_output = np.reshape(normalized_output, (-1, metadata['num_output_features']))
        normalized_input = np.reshape(normalized_input, newshape=(-1, sequence_length, num_input_features))

        # Compute the bin index
        #if 'bin_bounds' in metadata:
        #    output_val = normalized_output[0][0]  # There is only one output feature when bounds are used
        #    
        #    bin_index = 0
        #    num_bounds = len(metadata['bin_bounds'])
        #    for i in range(num_bounds - 1):
        #        curr_bound, next_bound = metadata['bin_bounds'][i], metadata['bin_bounds'][i+1]

        #        if i == 0 and output_val < curr_bound[0]:
        #            break
        #        if output_val > curr_bound[0] and output_val < next_bound[0]:
        #            break
        #        bin_index += 1

        #    normalized_output = [[bin_index]]

        if 'bin_means' in metadata:
            return {
                'inputs': np.array(normalized_input),
                'output': np.array(normalized_output),
                'bin_means': np.array(metadata['bin_means'])
            }

        return {
            'inputs': np.array(normalized_input),
            'output': np.array(normalized_output)
        }
