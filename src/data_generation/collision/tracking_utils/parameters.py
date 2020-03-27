import json
from os.path import exists
from datetime import datetime


class Parameters:

    def __init__(self, params_file: str):
        if not exists(params_file):
            raise ValueError('The parameters file {0} does not exist!'.format(params_file))

        if not params_file.endswith('.json'):
            raise ValueError('The parameters file must be a JSON.')

        with open(params_file, 'r') as f:
            params_dict = json.load(f)

        self.motion_model = params_dict['motion_model'].lower()
        self.frame_delay = int(params_dict['frame_delay'])

        self.object_start_location = tuple(params_dict['object_start_location'])
        self.object_size = float(params_dict['object_size'])
        self.object_type = params_dict['object_type']

        self.lamp_location = tuple(params_dict['lamp_location'])
        self.plane_dimensions = tuple(params_dict['plane_dimensions'])
        self.camera_locations = params_dict['camera_locations']
        self.motion_params = params_dict['motion_params']

        self.calibration_folder = params_dict['calibration_folder']
        self.calibration_iters = params_dict['calibration_iters']
        self.output_folder = params_dict['output_folder']
        self.evaluation_iters = params_dict['evaluation_iters']
        self.predictors = params_dict['predictors']

        self.camera_noise = params_dict['camera_noise']
        self.salt_pepper_prob = params_dict['salt_pepper_prob']
        self.motion_noise = params_dict['motion_noise']

        self.calibration_model = params_dict['calibration_model']
        self.calibration_params = params_dict['calibration_params']
        self.date = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
