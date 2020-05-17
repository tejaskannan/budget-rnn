import cv2
import os.path
import re
import numpy as np
import matplotlib.pyplot as plt
import time

from argparse import ArgumentParser
from multiprocessing import Pool
from functools import partial
from typing import Dict, Optional, List

from utils.constants import SAMPLE_ID, DATA_FIELDS, INPUTS, OUTPUT
from utils.file_utils import read_by_file_suffix, iterate_files
from utils.data_writer import NpzDataWriter, DataWriter


ID_REGEX = re.compile(r'.*/([0-9]+)\.webm')

ORIGINAL_ID = 'original-id'
ANGLES = [0, np.pi/4.0]
SCALES = [0.2, 0.3]
FILTER_SIZE = 4
NUM_CHUNKS = 4
HIST_SIZE = 5


def get_id(file_path: str) -> int:
    match = ID_REGEX.match(file_path)
    return int(match.group(1))


def create_data_label_dict(label_path: str) -> Dict[int, str]:
    result: Dict[int, str] = dict()

    for entry in read_by_file_suffix(label_path):
        sample_id, description = int(entry['id']), entry['template']
        result[sample_id] = description.replace('[', '').replace(']', '')

    return result


def get_label_for_sample(description: str, label_dict: Dict[str, str]) -> int:
    return int(label_dict[description])


def average_pool(array: np.ndarray, num_chunks: int) -> List[float]:
    features: List[float] = []

    x_stride = int(array.shape[0] / num_chunks)
    y_stride = int(array.shape[1] / num_chunks)

    for i in range(num_chunks):
        for j in range(num_chunks):
            x = i * x_stride
            y = j * y_stride

            chunk = array[x:x+x_stride, y:y+y_stride]
            avg = np.average(chunk)

            features.append(avg)

    return features


def get_features_from_image(image: np.ndarray, should_flip: bool) -> List[float]:
    if should_flip:
        image = np.flip(image, axis=1)

    features: List[float] = []

    start = time.time()

    # Apply Gabor Filters
    for angle in ANGLES:
        for scale in SCALES:
            kernel = cv2.getGaborKernel((FILTER_SIZE, FILTER_SIZE), scale, angle, 1.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered_img = cv2.filter2D(image, cv2.CV_8UC3, kernel)
            img_features = average_pool(filtered_img, num_chunks=NUM_CHUNKS)

            features.extend(img_features)

    # Add the color histogram
    for ch in range(image.shape[-1]):
        channel = image[:, :, ch]
        hist = cv2.calcHist([channel], channels=[0], mask=None, histSize=[HIST_SIZE], ranges=[0,256])
        hist_features = hist.reshape(-1).astype(float).tolist()

        features.extend(hist_features)
    
    return features


def downsample_image(image: np.ndarray, frac: float) -> np.ndarray:
    width = int(image.shape[1] * frac)
    height = int(image.shape[0] * frac)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def process_video(video_file: str, downsample_frac: float, frame_rate: int, max_num_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_file)    

    captured_frames = 0
    index = 0
    results: List[np.ndarray] = []

    ret, frame = cap.read()
    while ret and captured_frames < max_num_frames:
        if index % frame_rate == 0:
            results.append(downsample_image(frame, downsample_frac))
            captured_frames += 1

        ret, frame = cap.read()
        index += 1

    return results


def process_files(input_folder: str,
                  output_folder: str,
                  labels: Dict[str, str],
                  dataset_labels: Dict[int, str],
                  frame_rate: int,
                  max_num_frames: int,
                  downsample_frac: float,
                  chunk_size: int,
                  max_num_samples: Optional[int],
                  save_features: bool):
    data_files = iterate_files(input_folder, pattern=r'.*\.webm')

    if save_features:
        writer = DataWriter(output_folder, file_prefix='data', file_suffix='jsonl.gz', chunk_size=chunk_size, mode='w')
    else:
        writer = NpzDataWriter(output_folder, file_prefix='data', file_suffix='npz', chunk_size=chunk_size, sample_id_name=SAMPLE_ID, data_fields=DATA_FIELDS, mode='w')

    num_samples = 0
    new_sample_id = 0

    for file_path in data_files:

        sample_id = get_id(file_path)

        if sample_id not in dataset_labels:
            continue

        label = get_label_for_sample(dataset_labels[sample_id], labels)
        video_frames = process_video(file_path, downsample_frac, frame_rate, max_num_frames)

        if save_features:
            features_fn = partial(get_features_from_image, should_flip=False)

            with Pool() as pool:
                features = pool.map(features_fn, video_frames)
        else:
            features = video_frames

        data_dict = {
            SAMPLE_ID: new_sample_id,
            INPUTS: features,
            OUTPUT: label,
            ORIGINAL_ID: sample_id
        }
        writer.add(data_dict)

        new_sample_id += 1

        # If we are extracting features here, then use the reversed image too to up-sample the data
        if save_features:
            features_fn = partial(get_features_from_image, should_flip=True)
            with Pool() as pool:
                features = pool.map(features_fn, video_frames)

            data_dict = {
                SAMPLE_ID: new_sample_id,
                INPUTS: features,
                OUTPUT: label,
                ORIGINAL_ID: sample_id
            }
            writer.add(data_dict)

            new_sample_id += 1

        num_samples += 1

        if (max_num_samples is not None) and (num_samples >= max_num_samples):
            break

        if new_sample_id % chunk_size == 0:
            print(f'Completed {num_samples} samples.', end='\r')

    if num_samples >= chunk_size:
        print()

    writer.flush()
    print(f'Total of {new_sample_id} samples.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--labels-file', type=str, required=True)
    parser.add_argument('--dataset-labels', type=str, required=True)
    parser.add_argument('--frame-rate', type=int, required=True)
    parser.add_argument('--max-num-frames', type=int, required=True)
    parser.add_argument('--should-save-features', action='store_true')
    parser.add_argument('--chunk-size', type=int, default=10)
    parser.add_argument('--downsample-frac', type=float, default=1.0)
    parser.add_argument('--max-num-samples', type=int)
    args = parser.parse_args()

    # Validate files and parameters
    assert os.path.exists(args.input_folder), f'The input folder does not exist: {args.input_folder}'
    assert os.path.exists(args.labels_file), f'The labels file does not exist: {args.labels_file}'
    assert os.path.exists(args.dataset_labels), f'The dataset labels does does not exist: {args.dataset_labels}'

    assert args.downsample_frac > 0 and args.downsample_frac <= 1.0, 'The downsample fraction must be in (0, 1]'
    assert args.frame_rate > 0, 'Must have a positive frame rate'
    assert args.max_num_frames > 0, 'Must have a positive number of frames'

    data_labels = create_data_label_dict(args.dataset_labels)
    label_dict = read_by_file_suffix(args.labels_file)

    process_files(input_folder=args.input_folder,
                  output_folder=args.output_folder,
                  dataset_labels=data_labels,
                  labels=label_dict,
                  frame_rate=args.frame_rate,
                  max_num_frames=args.max_num_frames,
                  downsample_frac=args.downsample_frac,
                  chunk_size=args.chunk_size,
                  max_num_samples=args.max_num_samples,
                  save_features=args.should_save_features)
