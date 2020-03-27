import bpy
import os
import numpy as np
from typing import Any

from tracking_utils.position_utils import Point
from tracking_utils.file_utils import load_jsonl_gz


class Camera:

    def __init__(self, location: Point, rotation: Point, lens: int, name: str):
        self.__location = location
        self.__rotation = rotation
        self.__name = name
        self.__lens = lens
        self.__cam_object = None
        self.__cam = None

    @property
    def location(self) -> Point:
        return self.__location

    @property
    def rotation(self) -> Point:
        return self.__rotation

    @property
    def name(self) -> str:
        return self.__name

    @property
    def lens(self) -> int:
        return self.__lens

    @property
    def index(self) -> int:
        return self.__index

    def create(self, scene: Any):
        """
        Creates the camera in the given blender scene.
        """
        cam_data = bpy.data.cameras.new(name='cam')
        cam_ob = bpy.data.objects.new(name=self.name, object_data=cam_data)
        scene.objects.link(cam_ob)

        # Set location, rotation, and lens size
        cam_ob.location = tuple(self.location)
        cam_ob.rotation_euler = tuple(self.rotation)
        cam = bpy.data.cameras[cam_data.name]
        cam.lens = self.lens

        self.__cam_object = cam_ob
        self.__cam = cam

    def capture(self, scene: Any, folder: str, index: int):
        """
        Capture the current scene using this camera. The result
        is saved to the output folder.

        Args:
            scene: Blender scene
            folder: Output folder for which to save images
            index: Index of the current frame
        """
        scene.camera = self.__cam_object
        output_file = os.path.join(folder, '{0}-{1}.png'.format(self.name, index))
        scene.render.filepath = output_file
        bpy.ops.render.render(write_still=True)
