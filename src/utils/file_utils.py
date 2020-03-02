import re
from dpu_utils.utils import RichPath
from typing import Union


def to_rich_path(path: Union[str, RichPath]):
    if isinstance(path, str):
        return RichPath.create(path)
    return path


def extract_model_name(model_file: str) -> str:
    match = re.match(r'^model-([^\.]+)\.ckpt.*$', model_file)
    if not match:
        if model_file.startswith('model-'):
            return model_file[len('model-'):]
        return model_file
    return match.group(1)


