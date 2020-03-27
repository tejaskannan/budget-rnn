import bpy
from typing import Tuple

from blender.base import BlenderObject


class Plane(BlenderObject):

    def __init__(self,
                 location: Tuple[int, int, int],
                 dimensions: Tuple[int, int, int],
                 color: Tuple[float, float, float],
                 name: str):
        super().__init__(location=location, name=name, color=color)
        self.__dimensions = dimensions

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        return self.__dimensions

    def create(self, scene: bpy.types.Scene):
        bpy.ops.mesh.primitive_plane_add(location=self.location)  
        plane = bpy.context.object
        plane.dimensions = self.dimensions
        mat = bpy.data.materials.new('mat_{0}'.format(plane.name))
        mat.diffuse_color = self.color

        self.object = plane
