import numpy as np
import re
from dpu_utils.utils import RichPath
from enum import Enum, auto
from typing import Union, Dict, Any, DefaultDict, List, Generator
from collections import defaultdict
from more_itertools import ichunked

from utils.file_utils import to_rich_path


class DataSeries(Enum):
    TRAIN = auto()
    VALID = auto()
    TEST = auto()


class Dataset:

    def __init__(self, train_folder: Union[str, RichPath],
                       valid_folder: Union[str, RichPath],
                       test_folder: Union[str, RichPath]):
        self.data_folders = {
                DataSeries.TRAIN: to_rich_path(train_folder),
                DataSeries.VALID: to_rich_path(valid_folder),
                DataSeries.TEST: to_rich_path(test_folder)
        }

        match = re.match('^.+/(.+)/.+$', self.data_folders[DataSeries.TRAIN].path)
        self.dataset_name = match.group(1).replace('_', '-')

        # Load the dataset partitions
        self.dataset = {
            DataSeries.TRAIN: self.load_series(self.data_folders[DataSeries.TRAIN]),
            DataSeries.VALID: self.load_series(self.data_folders[DataSeries.VALID]),
            DataSeries.TEST: self.load_series(self.data_folders[DataSeries.TEST])
        }

    def tensorize(self, sample: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, np.ndarray]:
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

    def load_series(self, data_folder: RichPath) -> List[Dict[str, Any]]:
        """
        Loads data from the given series into memory. Only accepts data files
        which are stored as compressed .jsonl.gz files.

        Args:
            data_folder: The folder to load data from (TRAIN, VALID or TEST)
        Returns:
            A list of loaded data samples.
        """
        samples: List[Dict[str, Any]] = []
        for data_file in data_folder.iterate_filtered_files_in_dir('*.jsonl.gz'):
            for raw_sample in data_file.read_by_file_suffix():
                samples.append(self.process_raw_sample(raw_sample))

        return samples


    def minibatch_generator(self, series: DataSeries,
                                  batch_size: int,
                                  metadata: Dict[str, Any],
                                  should_shuffle: bool,
                                  drop_incomplete_batches: bool = False) -> Generator[DefaultDict[str, List[Any]], None, None]:
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

        # Shuffle the dataset if specified. Note that this shuffling mutates the original dataset.
        if should_shuffle:
            np.random.shuffle(data_series)

        for minibatch in ichunked(data_series, batch_size):
            # Turn minibatch into a feed dict
            feed_dict: DefaultDict[str, List[Any]] = defaultdict(list)
            num_samples = 0
            for sample in minibatch:
                tensorized_sample = self.tensorize(sample, metadata)
                for key, tensor in tensorized_sample.items():
                    feed_dict[key].append(tensor)
                num_samples += 1

            if drop_incomplete_batches and num_samples < batch_size:
                continue

            yield feed_dict
