import os
from typing import Iterable

from utils.file_utils import iterate_files
from .constants import CAMERA_FMT


def iterate_camera_images(image_folder: str) -> Iterable[str]:
    num_images = sum((1 for _ in iterate_files(image_folder, pattern=r'.*\.png')))

    for index in range(num_images):
        camera_file = CAMERA_FMT.format(index) + '.png'
        yield os.path.join(image_folder, camera_file)
