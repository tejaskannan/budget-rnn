import bpy
from typing import Tuple, Optional

from tracking_utils.position_utils import Point


class BlenderObject:

    def __init__(self,
                 location: Tuple[int, int, int],
                 name: str,
                 size: Optional[float] = None,
                 color: Optional[Tuple[float, float, float]] = None):
        self.__location = location
        self.__name = name
        self.__size = size
        self.__color = color
        self.object: Optional[bpy.types.Object] = None

    @property
    def location(self) -> Tuple[int, int, int]:
        return self.__location

    @property
    def name(self) -> str:
        return self.__name

    @property
    def size(self) -> Optional[float]:
        return self.__size

    @property
    def color(self) -> Optional[Tuple[float, float, float]]:
        return self.__color

    def set_location(self, new_location: Point):
        blender_obj = self.get_object()
        blender_obj.location = new_location
        blender_obj.keyframe_insert(data_path='location', index=-1)

    def get_object(self) -> bpy.types.Object:
        if self.object is None:
            raise ValueError('This object has not yet been created.')
        return self.object

    def create(self, scene: bpy.types.Scene):
        raise NotImplementedError()
