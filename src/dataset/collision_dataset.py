import numpy as np
from dataset.dataset import Dataset
from typing import Dict, Any

from utils.constants import INPUTS, OUTPUT, SAMPLE_ID, INPUT_NOISE, SMALL_NUMBER
from utils.constants import INPUT_SHAPE, INPUT_SCALER, OUTPUT_SCALER, NUM_OUTPUT_FEATURES
from data_generation.collision.tracking_utils.constants import COLLISION_FRAME


class CollisionDataset(Dataset):

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
            input_noise = np.random.normal(loc=0.0, scale=metadata[INPUT_NOISE], size=normalized_input.shape)
            normalized_input = np.array(normalized_input) + input_noise

        # If there is a collision, use the frame number as the class.
        # Otherwise, we use a distinct (no-frame) class. For convenience, this class
        # is equal to the sequence length.
        if abs(sample[OUTPUT]) < SMALL_NUMBER:
            output = [sequence_length]
        else:
            output = [sample[COLLISION_FRAME]]

        # Create output as a 1 x 1 matrix
        output = np.reshape(output, (-1, metadata[NUM_OUTPUT_FEATURES]))

        batch_dict = {
            INPUTS: normalized_input,
            OUTPUT: output,
            SAMPLE_ID: sample[SAMPLE_ID]
        }

        return batch_dict
