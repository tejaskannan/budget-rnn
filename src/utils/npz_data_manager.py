import numpy as np
import os

from typing import Iterable, List, Any, Dict

from .constants import DATA_FIELD_FORMAT, INDEX_FILE
from .file_utils import read_by_file_suffix, iterate_files


class NpzDataManager:

    def __init__(self, folder: str, sample_id_name: str, fields: List[str]):
        self._folder = folder
        assert os.path.exists(folder), f'The data folder {folder} does not exist!'

        self._sample_id_name = sample_id_name
        self._fields = fields

        self._is_loaded = False
        self._arrays: List[Any] = []
        self._array_lengths: List[int] = []
        self._ids: List[int] = []
        self._length = 0
        self._index: Dict[int, int] = dict()

    @property
    def folder(self):
        return self._folder

    @property
    def is_loaded(self):
        return self._is_loaded

    @property
    def length(self):
        return self._length

    @property
    def sample_id_name(self):
        return self._sample_id_name

    def load(self):
        # Prevent double-loading the dataset
        if self.is_loaded:
            return

        data_files = list(sorted(iterate_files(self.folder, pattern=r'.*\.npz')))

        if len(data_files) == 0:
            print('WARNING: No data files found.')
            return

        self._arrays = [np.load(data_file, mmap_mode='r') for data_file in data_files]
        self._array_lengths = [int(len(arr) / len(self._fields)) for arr in self._arrays]
        self._length = sum(self._array_lengths)
        self._ids = list(range(self._length))

        # Retrieve saved index or build sequential index if none given
        index_file = os.path.join(self.folder, INDEX_FILE)
        if os.path.exists(index_file):
            self._index = read_by_file_suffix(index_file)
            self._ids = list(sorted(self._index.keys()))
        else:
            for sample_id in self._ids:
                self._index[sample_id] = self._get_array_index(sample_id)

        self._is_loaded = True

    def shuffle(self):
        assert self._is_loaded, 'Must load the data before shuffling'
        np.random.shuffle(self._ids)

    def _get_array_index(self, sample_id: int) -> int:
        length_sum = 0
        for index, length in enumerate(self._array_lengths):
            length_sum += length
            if sample_id < length_sum:
                return index

        return len(self._array_lengths) - 1

    def iterate(self, should_shuffle: bool, batch_size: int) -> Iterable[Dict[str, Any]]:
        assert self._is_loaded, 'Must load the data before iterating'

        # Shuffle data if specified
        if should_shuffle:
            self.shuffle()

        batch: List[Dict[str, Any]] = []
        for sample_id in self._ids:
            arr_index = self._index[sample_id]
            arr = self._arrays[arr_index]

            # Load element into a dictionary
            element: Dict[str, Any] = dict()
            element[self.sample_id_name] = sample_id
            for field in self._fields:
                field_name = DATA_FIELD_FORMAT.format(field, sample_id)
                element[field] = arr[field_name]

            # Add to batch
            batch.append(element)

            # Emit elements
            if len(batch) == batch_size:
                for element in batch:
                    yield element
                batch = []

        # Emit any remaining elements
        for element in batch:
            yield element