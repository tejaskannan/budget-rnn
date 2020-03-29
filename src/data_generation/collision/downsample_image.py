import cv2
import os
import numpy as np
from typing import List
from argparse import ArgumentParser

from utils.constants import INPUTS, OUTPUT
from tracking_utils.constants import COLLISION_FRAME
from tracking_utils.file_utils import iterate_camera_images


FACTOR = 255.0


def downsample_images(image_folder: str, scale: float, label: int, collision_frame: int, output_file: str):
    images: List[np.ndarray] = []

    for image_path in iterate_camera_images(image_folder):
        # Read image
        image = cv2.imread(image_path)

        # Down-sample the image
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dims = (width, height)
        resized_image = cv2.resize(image, dims, interpolation=cv2.INTER_AREA)

        # Sum across RGB features to reduce space footprint
        compressed_image = resized_image

        # Normalize features
        resized_image = compressed_image / FACTOR
        resized_image = resized_image.astype(float)

        images.append(resized_image.tolist())

    # Save in an npz file
    np.savez_compressed(output_file, inputs=images, output=label, collision_frame=collision_frame)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--scale', type=float, required=True)
    parser.add_argument('--label', type=int, required=True)
    parser.add_argument('--collision-frame', type=int, required=True)
    parser.add_argument('--max-num', type=int)
    args = parser.parse_args()

    image_folder = args.folder

    num_files = len([t for t in os.listdir(image_folder) if t.endswith('.png')])
    if num_files == 0:
        for i, folder in enumerate(os.listdir(image_folder)):
            if args.max_num is not None and i < args.max_num:
                downsample_images(os.path.join(image_folder, folder), args.scale, args.label, args.collision_frame, args.output_file)
    else:
        downsample_images(image_folder, args.scale, args.label, args.collision_frame, args.output_file)
