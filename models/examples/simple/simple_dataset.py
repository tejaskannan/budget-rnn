import numpy as np
from dataset.dataset import Dataset
from typing import Dict, Any


class SimpleDataset(Dataset):

    def tensorize(self, sample: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, np.ndarray]:
        normalized_input = metadata['input_scaler'].transform([[sample['input']]])
        normalized_output = metadata['output_scaler'].transform([[sample['output']]])
 
        return {
            'input': normalized_input,
            'output': normalized_output
        }
