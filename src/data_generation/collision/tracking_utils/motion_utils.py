from enum import Enum, auto

from tracking_utils.parameters import Parameters
from tracking_utils.position_utils import Point
from motion_models.motion import MotionModel
from motion_models.constant import ConstantMotion
from motion_models.circle import CircleMotion
from motion_models.random import RandomMotion


class Mode(Enum):
    CALIBRATION = auto()
    EVAL = auto()


def get_motion_model(params: Parameters, mode: Mode) -> MotionModel:

    # The model and parameters can differ based on the execution mode (calibration or evaluation)
    model_name = params.motion_model if mode == Mode.EVAL else params.calibration_model
    model_params = params.motion_params if mode == Mode.EVAL else params.calibration_params

    if model_name == 'constant':
        return ConstantMotion(frame_delay=params.frame_delay,
                              positions=model_params['points'])
    elif model_name == 'circle':
        return CircleMotion(radius=model_params['radius'],
                            center=Point(*model_params['center']),
                            frame_delay=params.frame_delay,
                            traversal_frames=model_params['traversal_frames'],
                            noise=params.motion_noise)
    elif model_name == 'random':
        return RandomMotion(x_bound=tuple(model_params['x_bound']),
                            y_bound=tuple(model_params['y_bound']),
                            frame_delay=params.frame_delay,
                            random_factor=model_params['random_factor'])
    else:
        raise ValueError('Unknown model name: {0}'.format(model_name))


