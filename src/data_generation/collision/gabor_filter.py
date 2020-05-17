import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List

from feature_utils import average_pool, color_histogram, apply_gabor_filter, apply_laplace_filter


img = cv2.imread('scenes/collision_scenes/camera-0-0.png')

scale = 0.05
width = int(img.shape[1] * scale)
height = int(img.shape[0] * scale)
dims = (width, height)
img = cv2.resize(img, dims, interpolation=cv2.INTER_AREA)
img = cv2.GaussianBlur(img, (5, 5), 0)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(img, cmap='gray')
plt.show()

# Create Gabor Filter
num_angles = 4
sigmas = [0.1, 0.5]
for sigma in sigmas:
    filtered_image = apply_gabor_filter(gray_img, filter_size=4, scale=sigma, angle=0)

    print(average_pool(filtered_image, num_chunks=3))


    plt.imshow(filtered_image, cmap='gray')
    plt.show()
