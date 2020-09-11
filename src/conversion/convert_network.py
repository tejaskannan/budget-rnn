import numpy as np
import re
import os.path
from argparse import ArgumentParser
from collections import defaultdict
from typing import Callable, Set, Tuple, Dict, List, DefaultDict, Any, Optional

from utils.file_utils import read_by_file_suffix, extract_model_name, save_by_file_suffix
from utils.loading_utils import get_hyperparameters, get_metadata
from utils.sequence_model_utils import SequenceModelType
from utils.constants import EMBEDDING_NAME, OUTPUT_LAYER_NAME, STOP_PREDICTION, RNN_CELL_NAME, TRANSFORM_NAME
from utils.constants import SEQ_LENGTH, INPUT_SHAPE, NUM_OUTPUT_FEATURES, NUM_CLASSES
from conversion.layer_conversion import weight_matrix_conversion
from conversion.conversion_utils import parse_variable_name, create_constant, tensor_to_fixed_point
from conversion.conversion_utils import create_array, should_use_lea_ram, float_to_fixed_point
from controllers.model_controllers import CONTROLLER_PATH


def convert_network(model_path: str, model_parameters: Dict[str, np.ndarray], precision: int, is_msp: bool):
    # Extract the model meta-data and hyper-parameters
    hypers = get_hyperparameters(model_path)
    metadata = get_metadata(model_path)

    model_type = SequenceModelType[hypers.model_params['model_type'].upper()]

    # Holds a list of C variable declarations for each trainable parameter
    weight_variables: List[str] = []

    # Estimate LEA RAM consumption for model weights
    lea_ram_estimation = 0

    # Create C declarations for all trainable variables
    for var_name, var_value in model_parameters.items():
        layer_name, weight_type = parse_variable_name(var_name)

        c_declaration = None
        if layer_name in (EMBEDDING_NAME, STOP_PREDICTION, OUTPUT_LAYER_NAME, RNN_CELL_NAME):
            c_declaration = weight_matrix_conversion(layer_name, weight_type, var_value, precision=precision, is_msp=is_msp)
        else:
            raise ValueError('Unknown layer name: {0}'.format(layer_name))

        if isinstance(var_value, np.ndarray) and should_use_lea_ram(var_value):
            lea_ram_estimation += 2 * var_value.shape[0] * var_value.shape[1]

        weight_variables.append(c_declaration)

    print('LEA RAM Estimation: {0} bytes'.format(lea_ram_estimation))

    # Extract meta-data and hyper-parameters to create the computational graph
    state_size = hypers.model_params['state_size']
    num_outputs = hypers.model_params.get('num_outputs', 1)
    stride_length = hypers.model_params.get('stride_length', 1)
    rnn_cell_type = hypers.model_params['rnn_cell_type']

    seq_length = metadata[SEQ_LENGTH]
    num_input_features = metadata[INPUT_SHAPE][0]
    if hypers.model_params['output_type'] == 'multi_classification':
        num_output_features = metadata[NUM_CLASSES]
    else:
        num_output_features = metadata[NUM_OUTPUT_FEATURES]

    # Get thresholds for Sample RNNs
    thresholds = np.zeros(shape=(1, num_outputs))
    budgets = np.array([100])
    if model_type == SequenceModelType.SAMPLE_RNN:
        save_folder, model_file_name = os.path.split(model_path)
        model_name = extract_model_name(model_file_name)
        controller_path = os.path.join(save_folder, CONTROLLER_PATH.format(model_name))

        if os.path.exists(controller_path):
            controller_info = read_by_file_suffix(controller_path)
            thresholds = controller_info['thresholds']
            budgets = controller_info['budgets']

    with open('neural_network_parameters.h', 'w') as out_file:

        # Include necessary header files
        out_file.write('#include <stdint.h>\n')
        out_file.write('#include "math/matrix.h"\n\n')

        # Create header guard
        out_file.write('#ifndef NEURAL_NETWORK_PARAMS_GUARD\n')
        out_file.write('#define NEURAL_NETWORK_PARAMS_GUARD\n\n')

        # Create constants used during graph construction
        out_file.write(create_constant('FIXED_POINT_PRECISION', precision))
        out_file.write(create_constant('STATE_SIZE', state_size))
        out_file.write(create_constant('NUM_INPUT_FEATURES', num_input_features))
        out_file.write(create_constant('NUM_OUTPUT_FEATURES', num_output_features))
        out_file.write(create_constant('SEQ_LENGTH', seq_length))
        out_file.write(create_constant('NUM_OUTPUTS', num_outputs))
        out_file.write(create_constant('STRIDE_LENGTH', stride_length))
        out_file.write(create_constant('SAMPLES_PER_SEQ', int(seq_length / num_outputs)))
        out_file.write(create_constant('{0}_TRANSFORM'.format(rnn_cell_type.upper()), value=None))
        out_file.write(create_constant('IS_{0}'.format(model_type.name.upper()), value=None))

        if 'on_fraction' in hypers.model_params:
            on_fraction = float_to_fixed_point(hypers.model_params['on_fraction'], precision) 
            out_file.write(create_constant('ON_FRACTION', value=on_fraction))

        if is_msp:
            out_file.write(create_constant('IS_MSP', value=None))
            out_file.write(create_constant('VECTOR_COLS', value=2))
        else:
            out_file.write(create_constant('VECTOR_COLS', value=1))

        out_file.write('\n')

        thresholds = tensor_to_fixed_point(thresholds, precision=precision)
        thresholds_variable = create_array(thresholds, name='THRESHOLDS', dtype='int16_t')
        out_file.write(thresholds_variable + '\n')

        budgets_variable = create_array(budgets, name='BUDGETS', dtype='float')
        out_file.write(budgets_variable + '\n')

        out_file.write('\n\n'.join(weight_variables))

        out_file.write('\n#endif\n')


if __name__ == '__main__':
    parser = ArgumentParser('Compresses the neural network and converts the parameters into a C header file.')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--precision', type=int, required=True)
    parser.add_argument('--msp', action='store_true')
    args = parser.parse_args()

    assert args.precision > 0 and args.precision < 16, 'The precision must be in [1, 15]'

    model_parameters = read_by_file_suffix(args.model_path)
    convert_network(args.model_path, model_parameters, precision=args.precision, is_msp=args.msp)
