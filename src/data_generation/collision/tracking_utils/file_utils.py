import gzip
import codecs
import json
import pickle
from typing import Iterable, Any, Dict, List
from collections import OrderedDict


def save_jsonl_gz(data: Iterable[Any], file_path: str) -> None:
    assert file_path.endswith('.jsonl.gz'), 'Must provide a json lines file.'
    
    with gzip.GzipFile(file_path, 'wb') as f:
        writer = codecs.getwriter('utf-8')
        for element in data:
            writer(f).write(json.dumps(element))
            writer(f).write('\n')


def append_to_jsonl_gz(data_dict: Dict[str, Any], output_file: str):
    with gzip.GzipFile(output_file, 'a') as f:
        writer = codecs.getwriter('utf-8')
        json_str = json.dumps(data_dict) + '\n'
        writer(f).write(json_str)


def load_jsonl_gz(file_path: str) -> Iterable[Any]:
    assert file_path.endswith('.jsonl.gz'), 'Must provide a json lines file.'

    reader = codecs.getreader('utf-8')
    with gzip.open(file_path) as f:
        for line in reader(f):
            yield json.loads(line, object_pairs_hook=OrderedDict)


def save_as_pickle(data: Any, file_path: str) -> None:
    assert file_path.endswith('.pkl.gz'), 'Must provide a pickle file.'

    with gzip.GzipFile(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_as_pickle(file_path: str) -> Any:
    assert file_path.endswith('.pkl.gz'), 'Must provde a pickle file.'

    with gzip.open(file_path) as f:
        return pickle.load(f)
