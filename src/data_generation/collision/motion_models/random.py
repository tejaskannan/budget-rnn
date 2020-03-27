from random import random
from typing import Optional, Tuple

from motion_models.motion import MotionModel
from tracking_utils.position_utils import Position, Point, clip


class RandomMotion(MotionModel):

    def __init__(self, x_bound: Tuple[int, int], y_bound: Tuple[int, int],
                 frame_delay: int, random_factor: float = 1.0):
        super().__init__('random', frame_delay)
        assert x_bound[0] < x_bound[1], 'Must provide a valid x bound. Got: ' + str(x_bound)
        assert y_bound[0] < y_bound[1], 'Must provide a valid y bound. Got: ' + str(y_bound)

        self.__x_bound = x_bound
        self.__y_bound = y_bound
        self.__random_factor = random_factor

    def get_next_position(self, pos: Optional[Position] = None) -> Position:
        
        noise = self.__random_factor * random()
        
        if pos is None:
            next_point = Point(x=clip(noise, self.__x_bound),
                               y=clip(noise, self.__y_bound),
                               z=2)
            velocity = Point(x=0, y=0, z=0)
        else:
            next_point = Point(x=clip(noise + pos.location.x, self.__x_bound),
                               y=clip(noise + pos.location.y, self.__y_bound),
                               z=pos.location.z)

            dx = (next_point.x - pos.location.x) / self.frame_delay
            dy = (next_point.y - pos.location.y) / self.frame_delay
            dz = (next_point.z - pos.location.z) / self.frame_delay
            velocity = Point(x=dx, y=dy, z=dz)

        return Position(location=next_point, velocity=velocity)
