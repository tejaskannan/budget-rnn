from typing import Optional, List, Iterable

from motion_models.motion import MotionModel
from tracking_utils.position_utils import Position, Point


class ConstantMotion(MotionModel):

    def __init__(self, frame_delay: int, positions: Optional[List[Iterable[int]]]):
        super().__init__('constant', frame_delay)

        self.__positions = [Point(*p) for p in positions]
        self.__pos_index = 0

    def get_next_position(self, pos: Optional[Position] = None) -> Position:
        if self.__pos_index >= len(self.__positions):
            self.__pos_index = 0

        next_point = Point(*self.__positions[self.__pos_index])

        if pos is not None:
            dx = (next_point.x - pos.location.x) / self.frame_delay
            dy = (next_point.y - pos.location.y) / self.frame_delay
            dz = (next_point.z - pos.location.z) / self.frame_delay
            velocity = Point(x=dx, y=dy, z=dz)
        else:
            velocity = Point(x=0, y=0, z=0)

        self.__pos_index += 1

        return Position(location=next_point, velocity=velocity)
