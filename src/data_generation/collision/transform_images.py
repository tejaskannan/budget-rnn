import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from copy import deepcopy
from typing import List

from utils.constants import INPUTS
from utils.file_utils import read_by_file_suffix, iterate_files
from utils.data_writer import DataWriter


def reflect_images(images: List[List[List[List[float]]]], axis: int, show: bool) -> List[List[List[List[float]]]]:
    reflected: List[List[List[List[float]]]] = []
    for image in images:
        ref_image = np.flip(image, axis=axis).astype(float)

        if show:
            plt.imshow(np.array(image).astype(int), cmap='gray')
            plt.show()

            plt.imshow(ref_image.astype(int), cmap='gray')
            plt.show()

        reflected.append(ref_image.tolist())

    return reflected


def noisy_images(images: List[List[List[List[float]]]], std: float, show: bool) -> List[List[List[List[float]]]]:
    noisy: List[List[List[float]]] = []

    for image in images:
        height, width, channels = len(image), len(image[0]), len(image[0][0])
        noise = np.random.normal(loc=0.0, scale=std, size=(height, width, channels))

        noisy_img = np.clip(np.array(image) + noise, a_min=0.0, a_max=255.0).astype(float)

        if show:
            plt.imshow(noisy_img.astype(int))
            plt.show()

        noisy.append(noisy_img.tolist())

    return noisy


def transform_dataset(input_folder: str, output_folder: str, file_prefix: str, chunk_size: int):

    with DataWriter(output_folder, file_prefix=file_prefix, chunk_size=chunk_size, file_suffix='jsonl.gz') as writer:

        data_files = list(iterate_files(input_folder, pattern='.*jsonl.gz'))
        num_files = len(data_files)

        for file_index, data_file in enumerate(data_files):
            
            for sample_id, sample in enumerate(read_by_file_suffix(data_file)):
                writer.add(sample)

                # Include reflections to increase data diversity
                x_reflected_sample = deepcopy(sample)
                x_reflected_sample[INPUTS] = reflect_images(sample[INPUTS], axis=1, show=False)
                writer.add(x_reflected_sample)

                y_reflected_sample = deepcopy(sample)
                y_reflected_sample[INPUTS] = reflect_images(sample[INPUTS], axis=0, show=False)
                writer.add(y_reflected_sample)

                xy_reflected_sample = deepcopy(x_reflected_sample)
                xy_reflected_sample[INPUTS] = reflect_images(x_reflected_sample[INPUTS], axis=0, show=False)
                writer.add(xy_reflected_sample)

            print(f'Completed {file_index + 1}/{num_files} files.', end='\r')
        print()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--file-prefix', type=str, default='data')
    parser.add_argument('--chunk-size', type=int, default=500)
    args = parser.parse_args()

    assert os.path.exists(args.input_folder), f'The folder {args.input_folder} does not exist!'

    transform_dataset(input_folder=args.input_folder,
                      output_folder=args.output_folder,
                      file_prefix=args.file_prefix,
                      chunk_size=args.chunk_size)
