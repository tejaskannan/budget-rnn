import os

from utils.constants import TRAIN, VALID, TEST

from .dataset import Dataset
from .rnn_sample_dataset import RNNSampleDataset
from .collision_dataset import CollisionDataset
from .single_dataset import SingleDataset


def get_dataset(dataset_type: str, data_folder: str) -> Dataset:
    """
    Creates a dataset of the given type with the given base folder.
    This factory infers the training, validation and test folders.
    """
    dataset_type = dataset_type.lower()

    # Create data folders
    train_folder = os.path.join(data_folder, TRAIN)
    valid_folder = os.path.join(data_folder, VALID)
    test_folder = os.path.join(data_folder, TEST)

    if dataset_type == 'standard':
        return RNNSampleDataset(train_folder, valid_folder, test_folder)
    elif dataset_type == 'collision':
        return CollisionDataset(train_folder, valid_folder, test_folder)
    elif dataset_type == 'single':
        return SingleDataset(train_folder, valid_folder, test_folder)
    else:
        raise ValueError(f'Unknown dataset type: {dataset_type}')
