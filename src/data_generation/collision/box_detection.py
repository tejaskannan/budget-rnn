import cv2
import os
import numpy as np
import gzip
import json
from argparse import ArgumentParser
from typing import Dict, Any, List

from utils.constants import INPUTS, OUTPUT
from tracking_utils.constants import COLLISION_FRAME
from tracking_utils.file_utils import iterate_camera_images


FEATURES_PER_CONTOUR = 4


def detect_corners(path: str, num_contours: int) -> List[float]:
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray_blur, 35, 125)

    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    hierarchy = hierarchy[0]
    contour_areas = [cv2.contourArea(c) for c in contours]

    features = np.zeros(shape=(num_contours * FEATURES_PER_CONTOUR,))

    feature_index = 0
    seen_columns = set()
    seen_rows = set()

    if len(contour_areas) != len(contours):
        return None

    for _, contour in reversed(sorted(zip(contour_areas, contours))):
        if feature_index >= num_contours:
            break

        bounding_box = cv2.boundingRect(contour)

        # Ignore duplicate bounding boxes
        if bounding_box[0] in seen_columns and bounding_box[1] in seen_rows:
            continue

        for i in range(FEATURES_PER_CONTOUR):
            features[FEATURES_PER_CONTOUR * feature_index + i] = bounding_box[i]

        feature_index += 1
        seen_columns.add(bounding_box[0])
        seen_rows.add(bounding_box[1])

    # There fewer bounding boxes than anticipated, so copy previous features
    while feature_index < num_contours and feature_index > 0:
        # Copy in previous features
        for i in range(FEATURES_PER_CONTOUR):
            features[FEATURES_PER_CONTOUR * feature_index + i] = features[FEATURES_PER_CONTOUR * (feature_index - 1) + i]

        feature_index += 1

    # Explicitly convert to floats because not all file types can handle numpy data types
    return [float(x) for x in features]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--num-contours', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--collision-frame', type=int, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    args = parser.parse_args()

    num_contours = int(args.num_contours)
    label = int(args.label)

    should_write = True

    # Initialize the data dictionary
    data_dict = {
        INPUTS: [],
        OUTPUT: label,
        COLLISION_FRAME: args.collision_frame
    }

    for image_path in iterate_camera_images(args.folder):
        features = detect_corners(image_path, num_contours)

        if features is None:
            should_write = False
            break
        else:
            data_dict[INPUTS].append(features)

    if should_write:
        np.savez_compressed(args.output_file, **data_dict)
