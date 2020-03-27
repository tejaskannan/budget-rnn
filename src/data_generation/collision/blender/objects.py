import bpy
from typing import Tuple

from blender.base import BlenderObject


def create_target_object(object_type: str,
                         location: Tuple[int, int, int],
                         color: Tuple[float, float, float],
                         size: float,
                         index: int) -> BlenderObject:
    object_type_lower = object_type.lower()

    if object_type_lower == 'ball':
        return Ball(location=location,
                    color=color,
                    size=size,
                    name='{0}-{1}'.format(object_type_lower, index))
    raise ValueError('Unknown object typeL {0}'.format(object_type))


class Ball(BlenderObject):

    def create(self, scene: bpy.types.Scene):
        bpy.ops.mesh.primitive_uv_sphere_add(location=self.location,
                                             size=self.size)
        bpy.ops.object.shade_smooth()
        ball = bpy.context.object
        mat = bpy.data.materials.new('mat_{0}'.format(ball.name))
        mat.diffuse_color = self.color  # Red
        ball.data.materials.append(mat)

        self.object = ball


class Lamp(BlenderObject):

    def __init__(self,
                 location: Tuple[int, int, int],
                 name: str,
                 lamp_type: str):
        super().__init__(location=location, name=name)
        self.__lamp_type = lamp_type

    @property
    def lamp_type(self):
        return self.__lamp_type

    def create(self, scene: bpy.types.Scene):
        lamp_data = bpy.data.lamps.new(name=self.name.lower(), type=self.lamp_type)
        lamp_object = bpy.data.objects.new(name=self.name.upper(), object_data=lamp_data)  
        scene.objects.link(lamp_object)
        lamp_object.location = self.location

        self.object = lamp_object
