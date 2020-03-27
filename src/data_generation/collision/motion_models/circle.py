import math
import numpy as np
from typing import Optional, Union

from motion_models.motion import MotionModel
from tracking_utils.position_utils import Position, Point


class CircleMotion(MotionModel):

    def __init__(self, center: Point,
                 radius: float,
                 frame_delay: int,
                 traversal_frames: int,
                 noise: float = 0.0):
        super().__init__('circle', frame_delay, noise)
        self.__radius = radius
        self.__traversal_frames = traversal_frames
        self.__center = center
        self.__frame_counter = 0

        traversal_angle = (2 * math.pi) / (traversal_frames)
        sin_angle = math.sin(traversal_angle)
        cos_angle = math.cos(traversal_angle)
        self.transition_matrix = np.array([[cos_angle, sin_angle, 0.0], \
                                           [-1 * sin_angle, cos_angle, 0.0],
                                           [0.0, 0.0, 1.0]])

    @property
    def center(self) -> Point:
        return self.__center

    @property
    def radius(self) -> float:
        return self.__radius

    @property
    def start_pos(self) -> Point:
        return Point(self.center.x, self.center.y + self.radius, self.center.z)

    @property
    def traversal_frames(self):
        return self.__traversal_frames

    def get_next_position(self, pos: Optional[Position] = None) -> Position:
        if pos is None:
            return Position(location=self.start_pos, velocity=Point(0, 0, 0))

        # Scale and shift onto the unit circle
        xyz = np.array([pos.location.x, pos.location.y, pos.location.z])
        xyz_unit = np.array([(xyz[0] - self.center.x) / self.radius, (xyz[1] - self.center.y) / self.radius, xyz[2]])
        
        next_xyz_unit = self.transition_matrix.dot(xyz_unit)
        next_xyz = np.array([next_xyz_unit[0] * self.radius + self.center.x, next_xyz_unit[1] * self.radius + self.center.y, next_xyz_unit[2]])

        noisy_xyz = next_xyz + np.random.normal(loc=0.0, scale=self.noise, size=next_xyz.shape)

        new_pos = Point(x=noisy_xyz[0], y=noisy_xyz[1], z=pos.location.z)
        return Position(location=new_pos, velocity=pos.velocity)
