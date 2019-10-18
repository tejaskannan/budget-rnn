import numpy as np
from dataset.dataset import Dataset
from typing import Dict, Any


class RNNSampleDataset(Dataset):

    def tensorize(self, sample: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, np.ndarray]:

        num_input_features = len(sample['input_power'][0])
        sequence_length = len(sample['input_power'])
        input_sample = np.reshape(sample['input_power'], newshape=(-1, num_input_features))

        normalized_input = metadata['input_scaler'].transform(input_sample)
        normalized_output = metadata['output_scaler'].transform([[sample['output_power']]])

        normalized_input = np.reshape(normalized_input, newshape=(-1, sequence_length, num_input_features))
        return {
            'input_power': np.array(normalized_input),
            'output_power': np.array(normalized_output)
        }
