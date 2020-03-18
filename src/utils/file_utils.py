import re
import gzip
import json
import codecs
import pickle

from dpu_utils.utils import RichPath
from typing import Union, Optional, Iterable, Any

from .constants import MODEL_NAME_REGEX


def to_rich_path(path: Union[str, RichPath]):
    if isinstance(path, str):
        return RichPath.create(path)
    return path


def read_by_file_suffix(file_path: str) -> Any:
    if file_path.endswith('.jsonl.gz'):
        return read_jsonl_gz(file_path)
    elif file_path.endswith('.pkl.gz'):
        return read_pickle_gz(file_path)
    else:
        raise ValueError(f'Cannot read file: {file_path}')


def save_by_file_suffix(data: Any, file_path: str):
    if file_path.endswith('.jsonl.gz'):
        return save_jsonl_gz(data, file_path)
    elif file_path.endswith('.pkl.gz'):
        return save_pickle_gz(data, file_path)
    else:
        raise ValueError(f'Cannot save into file: {file_path}')


def save_jsonl_gz(data: Iterable[Any], file_path: str) -> None:
    assert file_path.endswith('.jsonl.gz'), 'Must provide a json lines gzip file.'
    
    with gzip.GzipFile(file_path, 'wb') as f:
        writer = codecs.getwriter('utf-8')
        for element in data:
            writer(f).write(json.dumps(element))
            writer(f).write('\n')


def read_jsonl_gz(file_path: str) -> Iterable[Any]:
    assert file_path.endswith('.jsonl.gz'), 'Must provide a json lines file.'

    reader = codecs.getreader('utf-8')
    with gzip.open(file_path) as f:
        for line in reader(f):
            yield json.loads(line, object_pairs_hook=OrderedDict)


def save_pickle_gz(data: Any, file_path: str) -> None:
    assert file_path.endswith('.pkl.gz'), 'Must provide a pickle file.'

    with gzip.GzipFile(file_path, 'wb') as f:
        pickle.dump(data, f)


def read_pickle_gz(file_path: str) -> Any:
    assert file_path.endswith('.pkl.gz'), 'Must provde a pickle file.'

    with gzip.open(file_path) as f:
        return pickle.load(f)


def extract_model_name(model_file: str) -> Optional[str]:
    match = re.match(MODEL_NAME_REGEX, model_file)
    if not match:
        return None
    return match.group(1)
