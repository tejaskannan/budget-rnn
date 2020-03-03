import re
from dpu_utils.utils import RichPath
from typing import Union, Optional

from .constants import MODEL_NAME_REGEX


def to_rich_path(path: Union[str, RichPath]):
    if isinstance(path, str):
        return RichPath.create(path)
    return path


def extract_model_name(model_file: str) -> Optional[str]:
    match = re.match(MODEL_NAME_REGEX, model_file)
    if not match:
        return None
    return match.group(1)
