import os
from typing import Iterable, List

from utils.file_utils import iterate_files
from .constants import CAMERA_FMT


def iterate_camera_images(image_folder: str, num_cameras: int) -> Iterable[List[str]]:
    num_images = sum((1 for _ in iterate_files(image_folder, pattern=r'.*\.png')))
    images_per_camera = int(num_images / num_cameras)

    for camera_index in range(num_cameras):

        camera_images: List[str] = []
        for image_index in range(images_per_camera):
            camera_file = '{0}-{1}.png'.format(CAMERA_FMT.format(camera_index), image_index)
            camera_images.append(os.path.join(image_folder, camera_file))

        yield camera_images
