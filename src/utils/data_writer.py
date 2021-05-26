import os
import re

from enum import Enum, auto
from typing import List, Any, Iterable, Dict

from utils.file_utils import save_by_file_suffix, iterate_files, make_dir
from utils.constants import INDEX_FILE, DATA_FIELD_FORMAT


class WriteMode(Enum):
    WRITE = auto()
    APPEND = auto()


class DataWriter:

    def __init__(self, output_folder: str, file_prefix: str, file_suffix: str, chunk_size: int, mode: str = 'w'):
        self._output_folder = output_folder
        self._file_prefix = file_prefix
        self._file_suffix = file_suffix
        self._chunk_size = chunk_size

        # Initialize the data list
        self._dataset: List[Any] = []

        # Create the output directory if necessary
        make_dir(self._output_folder)

        # Set the writing mode
        mode = mode.lower()
        if mode in ('w', 'write'):
            self._mode = WriteMode.WRITE
        elif mode in ('a', 'append'):
            self._mode = WriteMode.APPEND
        else:
            raise ValueError(f'Unknown writing mode: {mode}')

        # Set the initial file index
        self._file_index = 0
        if self._mode == WriteMode.APPEND:
            # Regex to extract index from existing files
            file_name_regex = re.compile(f'{file_prefix}([0-9]+)\.{file_suffix}')

            # Get index from all existing files
            for file_name in os.listdir(output_folder):
                match = file_name_regex.match(file_name)
                if match is not None:
                    index = int(match.group(1))
                    self._file_index = max(self._file_index, index + 1)

    @property
    def output_folder(self) -> str:
        return self._output_folder

    @property
    def file_prefix(self) -> str:
        return self._file_prefix
    
    @property
    def file_suffix(self) -> str:
        return self._file_suffix

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def file_index(self) -> int:
        return self._file_index

    def current_output_file(self) -> str:
        file_name = f'{self.file_prefix}{self.file_index:03}.{self.file_suffix}'
        return os.path.join(self.output_folder, file_name)

    def increment_file_index(self):
        self._file_index += 1

    def add(self, data: Any):
        self._dataset.append(data)
        if len(self._dataset) >= self.chunk_size:
            self.flush()

    def add_many(self, data: Iterable[Any]):
        for element in data:
            self.add(data)

    def flush(self):
        # Skip empty datasets
        if len(self._dataset) == 0:
            return

        save_by_file_suffix(self._dataset, self.current_output_file())
        self._dataset = []  # Reset the data list
        self.increment_file_index()

    def close(self):
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


class NpzDataWriter(DataWriter):

    def __init__(self, output_folder: str, file_prefix: str, file_suffix: str, chunk_size: int, sample_id_name: str, data_fields: List[str], mode: str):
        super().__init__(output_folder, file_prefix, file_suffix, chunk_size, mode)

        self._dataset: Dict[str, Any] = dict()
        self._index: Dict[int, int] = dict()
        self._sample_id_name = sample_id_name
        self._data_fields = data_fields
        self._length = 0

    def add(self, data: Any):
        assert isinstance(data, dict), 'Can only add dictionaries to npz files.'

        # Create fields with sample id
        sample_dict: Dict[str, Any] = dict()
        sample_id = data[self._sample_id_name]
        for field in self._data_fields:
            field_name = DATA_FIELD_FORMAT.format(field, sample_id)
            sample_dict[field_name] = data[field]

        self._dataset.update(**sample_dict)
        self._index[sample_id] = self.file_index
        self._length += 1

        if self._length >= self.chunk_size:
            self.flush()

    def flush(self):
        # Skip empty datasets
        if len(self._dataset) == 0:
            return

        # Save data and index
        save_by_file_suffix(self._dataset, self.current_output_file())
        save_by_file_suffix(self._index, os.path.join(self.output_folder, INDEX_FILE))
        self.increment_file_index()

        # Reset data. Keep the index as is--we always overwrite the index upon flushing
        self._dataset = dict()
        self._length = 0
