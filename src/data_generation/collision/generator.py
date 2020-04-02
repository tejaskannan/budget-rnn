import bpy
import os
import numpy as np
import json
import subprocess
import time
from mathutils import Vector
from typing import List, Dict, Optional, Any, Union

from motion_models.motion import MotionModel
from motion_models.circle import CircleMotion
from motion_models.collision import CollisionMotion
from motion_models.constant import ConstantMotion
from sensor.camera import Camera
from blender.worlds import Plane
from blender.objects import Lamp, create_target_object
from blender.base import BlenderObject
from tracking_utils.constants import *
from tracking_utils.parameters import Parameters
from tracking_utils.position_utils import Point
from tracking_utils.general_utils import get_random_point


def get_motion_model(params: Dict[str, Any], start_pos: List[float], end_pos: List[float], size: float, collision_iter: int, collision_pos: List[float]) -> MotionModel:

    model_name = params['motion_model'].lower()
    model_params = params['motion_params']

    if model_name == 'constant':
        return ConstantMotion(frame_delay=params['frame_delay'],
                              positions=model_params['points'])
    elif model_name == 'collision':
        return CollisionMotion(frame_delay=params['frame_delay'],
                               x_bound=model_params['x_bound'],
                               y_bound=model_params['y_bound'],
                               random_factor=model_params['random_factor'],
                               start_pos=Point(*start_pos),
                               collision_pos=Point(*collision_pos),
                               end_pos=Point(*end_pos),
                               collision_frame=collision_iter,
                               num_frames=params['iters'],
                               force_collision=model_params['force_collision'],
                               size=size)
    elif model_name == 'circle':
        return CircleMotion(radius=model_params['radius'],
                            center=Point(*model_params['center']),
                            frame_delay=params['frame_delay'],
                            traversal_frames=model_params['traversal_frames'],
                            noise=params['motion_noise'])
    else:
        raise ValueError('Unknown model name: {0}'.format(model_name))


def setup(world: BlenderObject,
          target_objects: List[BlenderObject],
          lamp: Lamp) -> List[bpy.types.Object]:
    """
    Initializes the world, target object and lighting.

    Args:
        world: Blender object representing the world
        target_object: The target object to track
        lamp: Lamp object used to light the world
    Returns:
        The underlying blender object for the target object
    """
    scene = bpy.context.scene

    # Clear everything
    scene.camera = None
    for obj in scene.objects:
        scene.objects.unlink(obj)

    # Build the world
    world.create(scene)
    lamp.create(scene)

    for target_object in target_objects:
        target_object.create(scene)

    return [target_object.get_object() for target_object in target_objects]


def create_camera(loc: Dict[str, List[int]], lens: int, index: int) -> Camera:
    """
    Initializes the camera sensors in the Blender world.

    Args:
        locations: List of dictionaries specifying the location
            and rotation of each sensor
        lens: Lens size
        index: Index number of this camera
    Returns:
        A list of the initialized camera objects
    """
    scene = bpy.context.scene

    cam = Camera(location=Point(*loc['location']),
                 rotation=Point(*loc['rotation']),
                 lens=lens,
                 name=CAMERA_FMT.format(index))
    cam.create(scene)

    return cam


def run_animation(target_objects: List[BlenderObject],
                  motion_models: List[MotionModel],
                  cameras: List[Camera],
                  output_folder: str,
                  max_iters: int):
    """
    Execute the animation and store results into the specified
    output folder.

    Args:
        target_object: Object to track
        motion_model: Model defining how the object moves in the world
        camera: Camera sensors
        output_folder: Folder in which to save all results
        max_iters: Maximum number of iterations of object motion
    """
    scene = bpy.context.scene
    frame_num = 0
    positions = [None for _ in range(len(target_objects))]

    # Set the start and ending frames
    scene.frame_start = 0
    scene.frame_end = (max_iters * motion_models[0].frame_delay) + 1

    # Start the animation
    bpy.ops.screen.animation_play()

    for i in range(max_iters):
        scene.frame_set(frame_num)

        for j, (target_object, motion_model) in enumerate(zip(target_objects, motion_models)):
            positions[j] = motion_model.get_next_position(positions[j])
            target_object.set_location(positions[j].location)

        # Capture images from each camera
        for camera in cameras:
            camera.capture(scene, index=i, folder=output_folder)

        frame_num += motion_model.frame_delay

    # Stop the animation
    bpy.ops.screen.animation_cancel()


def run_trial(params: Dict[str, Any], output_folder: str) -> int:
    """
    Main function to build world and run animation.

    Args:
        params_file: JSON file containing the run parameters
    """

    plane_color = tuple(np.random.uniform(low=0.5, high=0.8, size=(3,)))
    # Initialize the world objects
    plane = Plane(location=(0, 0, 0),
                  dimensions=params['plane_dimensions'],
                  color=plane_color,
                  name='plane')

    lamp = Lamp(location=params['lamp_location'],
                lamp_type='POINT',
                name='lamp')

    # Get the collision parameters for this trial
    model_params = params['motion_params']
    collision_iter = np.random.randint(low=2, high=params['iters'] - 1)
    collision_iters = np.random.randint(low=collision_iter, high=collision_iter + 2, size=(len(params['obj_configs']), ))
    collision_pos = get_random_point(model_params['collision_positions'])

    target_objects: List[BlenderObject] = []
    motion_models: List[MotionModel] = []
    for i, obj_config in enumerate(params['obj_configs']):
        start_location = get_random_point(obj_config['locations'])
        end_location = get_random_point(obj_config['locations'])

        color_index = np.random.randint(low=0, high=3)
        color = tuple([0 if i != color_index else 1 for i in range(3)])
        print(color)

        target_obj = create_target_object(object_type=obj_config['type'],
                                          location=start_location,
                                          size=obj_config['size'],
                                          color=color,
                                          index=i)
        target_objects.append(target_obj)

        motion_model = get_motion_model(params=params,
                                        start_pos=start_location,
                                        end_pos=end_location,
                                        size=obj_config['size'],
                                        collision_pos=collision_pos,
                                        collision_iter=collision_iters[i])

        motion_models.append(motion_model)

    # Initialize the world
    setup(world=plane,
          target_objects=target_objects,
          lamp=lamp)

    # Create camera sensors
    cameras: List[Camera] = []
    for camera_index, camera_location in enumerate(params['camera_locations']):
        camera = create_camera(loc=camera_location, lens=10, index=camera_index)
        cameras.append(camera)

    # Evaluation
    run_animation(target_objects, motion_models, cameras,
                  output_folder=output_folder,
                  max_iters=params['iters'])

    return collision_iter


def main(params_file: str):
    # Extract the parameters
    with open(params_file, 'r') as f:
        params = json.load(f)

    label = str(params['label'])
    output_folder = params['output_folder']
    output_file = params['features_file_prefix']
    scale = str(params['image_scale'])
    num_cameras = str(len(params['camera_locations']))

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for trial_index in range(params['trials']):
        collision_frame = str(run_trial(params, output_folder))

        # Sleep for a bit to ensure that the image files are saved on disk
        time.sleep(0.5)

        # Compute features (we call this in a subprocess to avoid library import issues)
        try:
            subprocess.check_call(['python3', 'downsample_image.py', '--folder', output_folder, '--output-file', output_file, '--label', label, '--collision-frame', collision_frame, '--scale', scale, '--num-cameras', num_cameras])
        except subprocess.CalledProcessError:
            pass

    # Save the parameters to logging purposes
    _, params_file_name = os.path.split(params_file)
    data_folder, _ = os.path.split(output_file)
    params_file_backup = os.path.join(data_folder, params_file_name)
    with open(params_file_backup, 'w') as f:
        json.dump(params, f)


if __name__ == '__main__':
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    for param_file in config['param_files']:
        main(param_file)
