import cv2
import os
import numpy as np
from typing import List
from argparse import ArgumentParser

from utils.constants import INPUTS, OUTPUT, SAMPLE_ID, DATA_FIELD_FORMAT
from utils.file_utils import read_by_file_suffix, save_by_file_suffix, make_dir, append_jsonl_gz
from tracking_utils.constants import COLLISION_FRAME, CAMERA_INDEX
from tracking_utils.file_utils import iterate_camera_images


METADATA_FILE = 'metadata.json'
DATA_FILE = '{0}{1:03d}.jsonl.gz'
FILE_INDEX = 'file_index'
CURRENT_LEN = 'current_len'
CHUNK_SIZE = 500


def downsample_images(image_folder: str, scale: float, label: int, collision_frame: int, num_cameras: int, output_file: str, should_save_images: bool):
    output_folder, file_prefix = os.path.split(output_file)

    # Get the file index, current file length, and sample id
    file_index = 0
    sample_id = 0
    current_length = 0
    metadata_path = os.path.join(output_folder, METADATA_FILE)
    metadata = dict()
    if os.path.exists(metadata_path):
        metadata = read_by_file_suffix(metadata_path)
        file_index = metadata[FILE_INDEX]  # Current file index
        sample_id = metadata[SAMPLE_ID]  # NEXT available sample ID
        current_length = metadata[CURRENT_LEN]  # Number of samples in the current file

    # Get the path of the current archive
    data_file_path = os.path.join(output_folder, DATA_FILE.format(file_prefix, file_index))

    for camera_index, camera_images in enumerate(iterate_camera_images(image_folder, num_cameras)):

        images: List[np.ndarray] = []

        # Get the path of the current archive
        data_file_path = os.path.join(output_folder, DATA_FILE.format(file_prefix, file_index))

        for i, image_path in enumerate(camera_images):
            # Read image
            image = cv2.imread(image_path)

            # Down-sample the image
            width = int(image.shape[1] * scale)
            height = int(image.shape[0] * scale)
            dims = (width, height)
            resized_image = cv2.resize(image, dims, interpolation=cv2.INTER_AREA)

            # Sum across RGB features to reduce space footprint
            compressed_image = resized_image

            # Convert to float to ensure compatability with all file types
            resized_image = compressed_image.astype(float)
            images.append(resized_image.tolist())

            if should_save_images:
                cv2.imwrite(f'downsampled/camera-{i}.png', compressed_image)

        # Append data to the current archive
        data_dict = {
            SAMPLE_ID: sample_id,
            INPUTS: images,
            OUTPUT: label,
            COLLISION_FRAME: collision_frame,
            CAMERA_INDEX: camera_index
        }
        append_jsonl_gz(data_dict, data_file_path)

        # Update metadata
        sample_id += 1
        current_length += 1
        if current_length >= CHUNK_SIZE:
            file_index += 1
            current_length = 0

        metadata[FILE_INDEX] = file_index
        metadata[SAMPLE_ID] = sample_id
        metadata[CURRENT_LEN] = current_length

    # Save metadata at the end
    save_by_file_suffix(metadata, metadata_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--scale', type=float, required=True)
    parser.add_argument('--label', type=int, required=True)
    parser.add_argument('--collision-frame', type=int, required=True)
    parser.add_argument('--num-cameras', type=int, required=True)
    parser.add_argument('--max-num', type=int)
    parser.add_argument('--save-images', action='store_true')
    args = parser.parse_args()

    # Make the output folder if necessary
    base_output_dir, _ = os.path.split(args.output_file)
    make_dir(base_output_dir)

    downsample_images(image_folder=args.folder,
                      scale=args.scale,
                      label=args.label,
                      collision_frame=args.collision_frame,
                      output_file=args.output_file,
                      num_cameras=args.num_cameras,
                      should_save_images=args.save_images)
