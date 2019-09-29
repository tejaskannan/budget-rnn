from dpu_utils.utils import RichPath
from typing import Union


def to_rich_path(path: Union[str, RichPath]):
    if isinstance(path, str):
        return RichPath.create(path)
    return path
