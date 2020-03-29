import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from utils.constants import INPUTS, OUTPUT
from tracking_utils.constants import COLLISION_FRAME
from tracking_utils.file_utils import iterate_camera_images


THRESHOLD = 12


def fast_detection(image_folder: str, label: int, collision_frame: int, num_points: int, show: bool, output_file: str):
    data_dict = {
        INPUTS: [],
        OUTPUT: label,
        COLLISION_FRAME: collision_frame
    }

    for image_file in iterate_camera_images(image_folder):

        img = cv2.imread(image_file)
        height, width, _channels = img.shape

        fast = cv2.FastFeatureDetector_create(threshold=THRESHOLD)
        keyPoints = fast.detect(img, None)

        if len(keyPoints) == 0:
            return

        points = list(sorted((kp.pt for kp in keyPoints)))[:num_points]
        normalized_points = [[point[0] / width, point[1] / height] for point in points]
        point_features = np.concatenate(normalized_points)

        if show:
            truncated_kp_indices = list(sorted(((kp.pt, i) for i, kp in enumerate(keyPoints))))[:num_points]
            truncated_kps = [keyPoints[j] for _, j in truncated_kp_indices]
            img2 = cv2.drawKeypoints(img, truncated_kps, None, color=(255, 0, 0))
            plt.imshow(img2)
            plt.show()

        pad_width = 2 * num_points - len(point_features)
        point_features = np.pad(point_features, pad_width=(0, pad_width), mode='constant', constant_values=0.0)

        if len(point_features) < 2 * num_points:
            return

        data_dict[INPUTS].append(point_features)

    np.savez_compressed(output_file, **data_dict)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--label', type=int, required=True)
    parser.add_argument('--collision-frame', type=int, required=True)
    parser.add_argument('--num-points', type=int, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    fast_detection(image_folder=args.folder,
                   label=args.label,
                   collision_frame=args.collision_frame,
                   num_points=args.num_points,
                   show=args.show,
                   output_file=args.output_file)
