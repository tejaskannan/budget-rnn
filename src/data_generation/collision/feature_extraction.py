import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Tuple, DefaultDict, Dict, Any
from sklearn.cluster import KMeans

from utils.constants import INPUTS, OUTPUT, SAMPLE_ID
from utils.file_utils import iterate_files, make_dir, save_by_file_suffix, read_by_file_suffix
from tracking_utils.constants import COLLISION_FRAME, CAMERA_INDEX
from tracking_utils.file_utils import iterate_camera_images
from feature_utils import apply_gabor_filter, average_pool, apply_laplace_filter


THRESHOLD = 10
FEATURES_PER_CONTOUR = 4


def gist_features(image_sample: Dict[str, Any],
                  filter_size: int,
                  scales: List[float],
                  angles: List[float],
                  num_chunks: int,
                  collision_cameras: List[Dict[str, Any]],
                  no_collision_cameras: List[Dict[str, Any]],
                  should_show: bool) -> Dict[str, Any]:
    data_dict = {
        INPUTS: [],
        OUTPUT: image_sample[OUTPUT],
        COLLISION_FRAME: image_sample[COLLISION_FRAME],
        SAMPLE_ID: image_sample[SAMPLE_ID],
        CAMERA_INDEX: image_sample[CAMERA_INDEX]
    }

    # Get features from camera angle
    if image_sample[OUTPUT] == 0:
        camera = no_collision_cameras[image_sample[CAMERA_INDEX]]
    else:
        camera = collision_cameras[image_sample[CAMERA_INDEX]]

    camera_features = camera['location'] + camera['rotation']

    # Process each individual image
    for image_matrix in image_sample[INPUTS]:
        # Collect the image
        img = np.array(image_matrix).astype(np.uint8)

        # Apply the Laplace filter and average pool the result
        laplace_filtered = apply_laplace_filter(img)
        laplace_features = average_pool(laplace_filtered, num_chunks=num_chunks)

        if should_show:
            plt.imshow(laplace_filtered)
            plt.show()

        # Apply the gabor filter and average pool the result
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gabor_features: List[float] = []
        for scale in scales:
            for angle in angles:
                filtered_image = apply_gabor_filter(grayscale_img, filter_size=filter_size, scale=scale, angle=angle)

                if should_show:
                    plt.imshow(filtered_image)
                    plt.show()

                filter_features = average_pool(filtered_image, num_chunks=num_chunks) 
                gabor_features.extend(filter_features)

        features = list(np.concatenate([laplace_features, gabor_features, camera_features]).astype(float))
        data_dict[INPUTS].append(features)

    return data_dict


def fast_detection(image_sample: Dict[str, Any], num_points: int, should_show: bool) -> Dict[str, Any]:
    data_dict = {
        INPUTS: [],
        OUTPUT: image_sample[OUTPUT],
        COLLISION_FRAME: image_sample[COLLISION_FRAME],
        SAMPLE_ID: image_sample[SAMPLE_ID],
        CAMERA_INDEX: image_sample[CAMERA_INDEX]
    }

    for image_matrix in image_sample[INPUTS]:
        # Collect the image
        img = np.array(image_matrix).astype(np.uint8)
        height, width, _channels = img.shape

        # Detect key points using the FAST algorithm
        fast = cv2.FastFeatureDetector_create(threshold=THRESHOLD)
        key_points = fast.detect(img, None)

        if len(key_points) == 0:
            return None

        if len(key_points) >= num_points:
            points = np.vstack([kp.pt for kp in key_points])
            clusters = KMeans(n_clusters=num_points).fit(points)
            points_to_keep = [(pt[0], pt[1]) for pt in clusters.cluster_centers_]
        else:
            points_to_keep = [kp.pt for kp in key_points]

        # Create point features
        normalized_points = [[point[0] / width, point[1] / height] for point in sorted(points_to_keep)]
        point_features = np.concatenate(normalized_points)

        if should_show:
            radius = key_points[0].size
            truncated_kps = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=radius) for pt in points_to_keep]

            img2 = cv2.drawKeypoints(img, truncated_kps, None, color=(0, 255, 0))
            plt.imshow(img2)
            plt.show()

        # Pad features if necessary
        pad_width = 2 * num_points - len(point_features)
        point_features = np.pad(point_features, pad_width=(0, pad_width), mode='constant', constant_values=0.0)

        if len(point_features) < 2 * num_points:
            return

        # Convert to floats because numpy arrays are not JSON serializable
        point_features = [float(x) for x in point_features]
        data_dict[INPUTS].append(point_features)

    return data_dict


def box_detection(image_sample: Dict[str, Any], num_points: int, should_show: bool):
    data_dict = {
        INPUTS: [],
        OUTPUT: image_sample[OUTPUT],
        COLLISION_FRAME: image_sample[COLLISION_FRAME],
        SAMPLE_ID: image_sample[SAMPLE_ID],
        CAMERA_INDEX: image_sample[CAMERA_INDEX]
    }

    for image_matrix in image_sample[INPUTS]:
        img = np.array(image_matrix).astype(np.uint8)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
        edges = cv2.Canny(img_blur, 35, 125)

        contours, heirarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        contour_areas = [cv2.contourArea(c) for c in contours]
        contour_features: List[np.ndarray] = []

        seen_boxes = set()

        feature_index = 0
        for _, contour in reversed(sorted(zip(contour_areas, contours))):
            if feature_index >= num_points:
                break

            bounding_box = cv2.boundingRect(contour)
            if bounding_box in seen_boxes:
                continue

            contour_features.append(bounding_box) 

            seen_boxes.add(bounding_box)
            feature_index += 1

        sorted_features = list(sorted(contour_features))
        features = np.concatenate(sorted_features)

        pad_width = 4 * num_points - len(features)
        features = np.pad(features, pad_width=(0, pad_width), mode='constant', constant_values=0.0)

        # Convert to float because numpy data types are not JSON serializable
        float_features = [float(x) for x in features]
        data_dict[INPUTS].append(float_features)

        if should_show:
            img2 = img
            for box in seen_boxes:
                x, y, w, h = box
                img2 = cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 0), 1)
            plt.imshow(img2)
            plt.show()

    return data_dict


def process_directory(input_folder: str,
                      file_prefix: str,
                      output_folder: str,
                      params: Dict[str, Any],
                      collision_cameras: List[Dict[str, Any]],
                      no_collision_cameras: List[Dict[str, Any]],
                      should_show: bool):

    data_files = list(iterate_files(input_folder, pattern=r'.*jsonl.gz'))
    make_dir(output_folder)

    # Get the extraction mode
    mode = params['mode'].lower()
    assert mode in ('fast', 'box', 'gist'), 'Invalid mode. Must be one of: fast, box, gist.'

    num_files = len(data_files)
    for i, data_file in enumerate(data_files):
        # Get the output file name
        _, file_name = os.path.split(data_file)
        output_path = os.path.join(output_folder, file_name)

        dataset: List[Dict[str, Any]] = []
        for sample in read_by_file_suffix(data_file):

            if mode == 'fast':
                features = fast_detection(sample, params['num_points'], should_show)
            elif mode == 'box':
                features = box_detection(sample, params['num_points'], should_show)
            elif mode == 'gist':
                features = gist_features(image_sample=sample,
                                         filter_size=params['filter_size'],
                                         scales=params['scales'],
                                         angles=params['angles'],
                                         num_chunks=params['num_chunks'],
                                         collision_cameras=collision_cameras,
                                         no_collision_cameras=no_collision_cameras,
                                         should_show=should_show)

            if features is not None:
                dataset.append(features)

        save_by_file_suffix(dataset, output_path)
 
        print(f'Completed {i+1}/{num_files} files.', end='\r')
    print()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--file-prefix', type=str, default='data')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--collision-params', type=str, required=True)
    parser.add_argument('--no-collision-params', type=str, required=True)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    assert os.path.exists(args.params), f'The file {args.params} does not exist!'
    params = read_by_file_suffix(args.params)

    assert os.path.exists(args.collision_params), f'The file {args.collision_params} does not exist!'
    collision_params = read_by_file_suffix(args.collision_params)
    collision_cameras = collision_params['camera_locations']

    assert os.path.exists(args.no_collision_params), f'The file {args.no_collision_params} does not exist!'
    no_collision_params = read_by_file_suffix(args.no_collision_params)
    no_collision_cameras = no_collision_params['camera_locations']

    process_directory(input_folder=args.input_folder,
                      file_prefix=args.file_prefix,
                      output_folder=args.output_folder,
                      params=params,
                      collision_cameras=collision_cameras,
                      no_collision_cameras=no_collision_cameras,
                      should_show=args.show)
