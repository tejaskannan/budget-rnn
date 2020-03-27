import math
import numpy as np
from collections import namedtuple
from typing import Tuple

Prediction = namedtuple('Prediction', ['global_pred', 'sensor_preds', 'num_communicated'])
Point = namedtuple('Point', ['x', 'y', 'z'])
Position = namedtuple('Position', ['location', 'velocity'])
Homography = namedtuple('Homography', ['h', 'h_inv'])

class BoundingBox:

    __slots__ = ['top_left', 'top_right', 'bottom_left', 'bottom_right']

    def __init__(self,
                 top_left: Point,
                 top_right: Point,
                 bottom_left: Point,
                 bottom_right: Point):
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
    
    def get_center(self) -> Point:
        return Point(x=self.top_left.x + int((self.top_right.x - self.top_left.x) / 2),
                     y=self.top_left.y + int((self.bottom_left.y - self.top_left.y) / 2),
                     z=1)

    def to_tuple(self) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        return ((self.top_left.x, self.top_left.y),
                (self.top_right.x, self.top_right.y),
                (self.bottom_left.x, self.bottom_left.y),
                (self.bottom_right.x, self.bottom_right.y))

    def to_numpy_array(self) -> np.ndarray:
        points_list = list(self.to_tuple())
        arr = np.array(points_list)
        return arr

    def __str__(self) -> str:
        return 'Top Left: {0}, Top Right: {1}, Bottom Left: {2}, Bottom Right: {3}'\
                    .format(self.top_left, self.top_right, self.bottom_left, self.bottom_right)


def euclidean_distance(p1: Point, p2: Point) -> float:
    dx = (p1.x - p2.x)**2
    dy = (p1.y - p2.y)**2
    dz = (p1.z - p2.z)**2
    return math.sqrt(dx + dy + dz)


def clip(v: float, bound: Tuple[int, int]) -> float:
    if v < bound[0]:
        return float(bound[0])
    elif v > bound[1]:
        return float(bound[1])
    return v


def add_noise(point: Point, variance: float) -> Point:
    noise = np.random.normal(loc=point, scale=variance, shape=(3,))
    return Point(x=point.x + noise[0], y=point.y + noise[1], z=point.z + noise[2])
