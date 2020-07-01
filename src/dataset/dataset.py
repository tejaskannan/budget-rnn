import numpy as np
import re
from enum import Enum, auto
from typing import Union, Dict, Any, DefaultDict, List, Generator, Iterable
from collections import defaultdict
from more_itertools import ichunked

from utils.constants import SAMPLE_ID, DATA_FIELDS, OUTPUT
from .data_manager import get_data_manager


class DataSeries(Enum):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


DEFAULT_BATCH_SIZE = 100


class Dataset:

    def __init__(self, train_folder: str, valid_folder: str, test_folder: str):
        self.data_folders = {
                DataSeries.TRAIN: train_folder,
                DataSeries.VALID: valid_folder,
                DataSeries.TEST: test_folder
        }

        # Extract the name from the given folder
        match = re.match('^.+/(.+)/.+/.+$', self.data_folders[DataSeries.TRAIN])
        self.dataset_name = match.group(1).replace('_', '-')

        # Create data managers for each partition
        self.dataset = {
            DataSeries.TRAIN: get_data_manager(self.data_folders[DataSeries.TRAIN], SAMPLE_ID, DATA_FIELDS),
            DataSeries.VALID: get_data_manager(self.data_folders[DataSeries.VALID], SAMPLE_ID, DATA_FIELDS),
            DataSeries.TEST: get_data_manager(self.data_folders[DataSeries.TEST], SAMPLE_ID, DATA_FIELDS)
        }

    def tensorize(self, sample: Dict[str, Any], metadata: Dict[str, Any], is_train: bool) -> Dict[str, np.ndarray]:
        pass

    def process_raw_sample(self, raw_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transforms a raw sample into a data sample to be fed into the model.
        The default behavior is the identity function.

        Args:
            raw_sample: The raw data sample loaded directly from a data file.
        Returns:
            A transformed data sample to be fed into the model.
        """
        return raw_sample

    def iterate_series(self, series: DataSeries) -> Iterable[Dict[str, Any]]:
        """
        Returns an iterator over all samples in the given series
        """
        data_series = self.dataset[series]
        if not data_series.is_loaded:
            data_series.load()

        return data_series.iterate(should_shuffle=False, batch_size=DEFAULT_BATCH_SIZE)

    def minibatch_generator(self, series: DataSeries,
                                  batch_size: int,
                                  metadata: Dict[str, Any],
                                  should_shuffle: bool,
                                  drop_incomplete_batches: bool = False,
                                  order_by_output: bool = False) -> Generator[DefaultDict[str, List[Any]], None, None]:
        """
        Generates minibatches for the given dataset. Each minibatch is expressed as a feed dict with string keys. These keys
        must be translated to placeholder tensors before passing the dictionary as an input to Tensorflow.

        Args:
            series: The series to generate batches for.
            batch_size: The minibatch size.
            metadata: Metadata used during tensorization
            should_shuffle: Whether the data samples should be shuffled
            drop_incomplete_batches: Whether incomplete batches should be omitted. This usually applies
                exclusively to the final minibatch.
        Returns:
            A generator a feed dicts, each one representing an entire minibatch.
        """
        data_series = self.dataset[series]

        # Load dataset if needed
        if not data_series.is_loaded:
            data_series.load()

        # Create iterator over the data
        if order_by_output:
            data_iterator = data_series.order_by_field(field=OUTPUT, min_length=5, max_length=15)
        else:
            data_iterator = data_series.iterate(should_shuffle=should_shuffle, batch_size=batch_size)

        # Set training flag
        is_train = series == DataSeries.TRAIN

        # Generate minibatches
        for minibatch in ichunked(data_iterator, batch_size):
            # Turn minibatch into a feed dict
            feed_dict: DefaultDict[str, List[Any]] = defaultdict(list)
            num_samples = 0
            for sample in minibatch:
                tensorized_sample = self.tensorize(sample, metadata, is_train=is_train)

                # Ensure that there are no NoneType or NaN values in the tensorized sample
                should_include = True
                for tensor in tensorized_sample.values():
                    if isinstance(tensor, list) or isinstance(tensor, np.ndarray):
                        tensor_array = np.array(tensor)

                        if np.any(np.isnan(tensor_array)) or np.any(tensor_array == None):
                            should_include = False
                    else:
                        if tensor is None or np.isnan(tensor):
                            should_include = False

                    if not should_include:
                        break

                # Only include validated samples
                if should_include:
                    for key, tensor in tensorized_sample.items():
                        feed_dict[key].append(tensor)
                    num_samples += 1

            if drop_incomplete_batches and num_samples < batch_size:
                continue

            yield feed_dict

    def close(self):
        for data_series in self.dataset.values():
            data_series.close()
