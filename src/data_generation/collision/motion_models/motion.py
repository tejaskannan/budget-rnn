import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from tracking_utils.position_utils import Position, Point


class MotionModel(ABC):

    def __init__(self, name: str, frame_delay: int, noise: float = 0.0):
        self.__name = name
        self.__frame_delay = frame_delay
        self.__noise = noise
        self.transition_matrix = np.identity

    @property
    def name(self):
        return name

    @property
    def frame_delay(self):
        return self.__frame_delay

    @property
    def noise(self):
        return self.__noise

    @property
    def start_pos(self) -> Point:
        return Point(x=1.0, y=0.0, z=0.0)

    @abstractmethod
    def get_next_position(self, pos: Optional[Position] = None) -> Position:
        """
        Returns the position after `frame_delay` frames when
        starting from the current position.
        """
        pass
