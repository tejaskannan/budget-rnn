import numpy as np
from typing import Optional, Tuple

from motion_models.motion import MotionModel
from tracking_utils.position_utils import Position, Point, clip
from tracking_utils.general_utils import softmax
from tracking_utils.constants import SMALL_NUMBER, COLLISION_NOISE_FACTOR


class CollisionMotion(MotionModel):

    def __init__(self, x_bound: Tuple[int, int], y_bound: Tuple[int, int],
                 frame_delay: int, random_factor: float, start_pos: Point, collision_pos: Point,
                 end_pos: Point, collision_frame: int, num_frames: int, force_collision: bool, size: float):
        super().__init__('collision', frame_delay)
        assert x_bound[0] < x_bound[1], 'Must provide a valid x bound. Got: ' + str(x_bound)
        assert y_bound[0] < y_bound[1], 'Must provide a valid y bound. Got: ' + str(y_bound)

        self.__x_bound = x_bound
        self.__y_bound = y_bound
        self.__random_factor = random_factor
        self.__start_pos = start_pos
        self.__collision_pos = collision_pos
        self.__end_pos = end_pos
        self.__num_frames = num_frames
        self.__collision_frame = collision_frame
        
        if collision_frame > 1:
            self.__steps = softmax(np.random.normal(loc=0.0, scale=1.0, size=(collision_frame - 1,)))
        else:
            self.__steps = np.array([1])

        self.__t = 0.0
        self.__iter = 0
        self.__force_collision = force_collision
        self.__size = size
        self.__has_collided = False

    @property
    def collision_frame(self) -> int:
        return self.__collision_frame

    def get_next_position(self, pos: Optional[Position] = None) -> Position:
        if abs(1.0 - self.__t) > SMALL_NUMBER or not self.__force_collision:
            noise = np.random.normal(loc=0.0, scale=self.__random_factor, size=(2,))
        else:
            noise = np.random.uniform(low=0.1, high=self.__size * COLLISION_NOISE_FACTOR, size=(2,))

        if pos is None:
            self.__t += self.__steps[0]
            self.__iter += 1
            return Position(location=self.__start_pos, velocity=Point(x=0, y=0, z=0))

        if self.__has_collided or not self.__force_collision:
            prev = self.__collision_pos if self.__force_collision else self.__start_pos

            x = self.__t * self.__end_pos.x + (1.0 - self.__t) * prev.x + noise[0]
            y = self.__t * self.__end_pos.y + (1.0 - self.__t) * prev.y + noise[1]
        else:
            x = self.__t * self.__collision_pos.x + (1.0 - self.__t) * self.__start_pos.x + noise[0]
            y = self.__t * self.__collision_pos.y + (1.0 - self.__t) * self.__start_pos.y + noise[1]

        next_point = Point(x=clip(x, self.__x_bound),
                           y=clip(y, self.__y_bound),
                           z=pos.location.z)

        dx = (next_point.x - pos.location.x) / self.frame_delay
        dy = (next_point.y - pos.location.y) / self.frame_delay
        dz = (next_point.z - pos.location.z) / self.frame_delay
        velocity = Point(x=dx, y=dy, z=dz)

        if self.__iter < len(self.__steps):
            self.__t += self.__steps[self.__iter]
            self.__iter += 1
        elif self.__iter == len(self.__steps):
            self.__steps = softmax(np.random.normal(loc=0.0, scale=1.0, size=(self.__num_frames - self.__collision_frame + 1, )))
            self.__t = 0
            self.__iter = 0
            self.__has_collided = True
        else:
            self.__t += 1.0

        return Position(location=next_point, velocity=velocity)
