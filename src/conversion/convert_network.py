import numpy as np
import re
import os.path
from argparse import ArgumentParser
from collections import defaultdict
from typing import Callable, Set, Tuple, Dict, List, DefaultDict, Any, Optional
from functools import partial

from utils.hyperparameters import HyperParameters
from utils.file_utils import read_by_file_suffix, extract_model_name, save_by_file_suffix
from utils.constants import SMALL_NUMBER, BIG_NUMBER, HYPERS_PATH, METADATA_PATH


BIAS = 'BIAS'
KERNEL = 'KERNEL'
MATRIX = 'MAT'
FUSION = 'FUSION'
COMBINE_STATES = 'combine-states'
LAYER_REGEX = re.compile('.*/.*-([0-9]+).*')
RNN_WEIGHT_REGEX = re.compile('.*/(rnn-cell).*-cell-([0-9+])-(.*):.*')
MULTI_RNN_CELL_REGEX = re.compile('.*transform-(layer-cell)-([0-9]+)/([^/]+)/([^:]+)')
INDEX_REGEX = re.compile('.*([0-9]+).*')
LEVEL_REGEX = re.compile('.*level[_-]([0-9]+).*')

NAME_FORMAT = '{0}_{1}_{2}'
LAYER_ARRAY_FORMAT = 'static void * {0}[{1}] = {{ {2} }};'
OUTPUT_NAME_REGEX = re.compile('.*/output-.*(hidden)')
C_VAR_FORMAT = 'static {0} {1} = {2};'
ARRAY_FORMAT = '{0}[{1}]'


def create_matrix(name: str, dim0: int, dim1: int) -> str:
    var_name = '{0}_{1}_VAR'.format(name, MATRIX)
    ptr_name = '{0}_{1}'.format(name, MATRIX)

    struct_assignment = ', '.join([name, str(dim0), str(dim1)])

    code_list: List[str] = []
    code_list.append('static matrix {0} = {{ {1} }};'.format(var_name, struct_assignment))
    code_list.append(C_VAR_FORMAT.format('matrix *', ptr_name, '&' + var_name))

    return '\n'.join(code_list)


def get_hypers(model_path: str) -> HyperParameters:
    save_folder, model_file = os.path.split(model_path)

    model_name = extract_model_name(model_file)
    hypers_name = HYPERS_PATH.format(model_name)
    return HyperParameters.create_from_file(os.path.join(save_folder, hypers_name))


def get_metadata(model_path: str) -> Dict[str, Any]:
    save_folder, model_file = os.path.split(model_path)

    model_name = extract_model_name(model_file)
    metadata_name = METADATA_PATH.format(model_name)
    metadata = read_by_file_suffix(os.path.join(save_folder, metadata_name))
    return metadata['metadata']


def seq_to_str(arr: List[Any]) -> str:
    return '{{ {0} }}'.format(','.join(map(str, arr)))


def collect_dense_layer(variable_names: List[str], layer_name: str, activation: str, activate_final: bool) -> List[Dict[str, str]]:
    layer_dict: DefaultDict[int, Dict[str, str]] = defaultdict(dict)
    output_dict = dict()

    for variable_name in variable_names:
        var_name = variable_name.lower()

        if layer_name not in var_name:
            continue

        match = INDEX_REGEX.match(variable_name)
        index = int(match.group(1))

        var_dict = layer_dict[index] if 'hidden' in var_name else output_dict

        if 'kernel' in var_name.lower():
            var_dict['kernel'] = variable_name
        elif 'bias' in var_name.lower():
            var_dict['bias'] = variable_name

        if 'hidden' in var_name or activate_final:
            var_dict['activation'] = activation
        else:
            var_dict['activation'] = 'linear'

    layer_list: List[Dict[str, str]] = []
    for _, var_dict in sorted(layer_dict.items()):
        layer_list.append(var_dict)

    if len(output_dict) > 0:
        layer_list.append(output_dict)

    return layer_list


def collect_recurrent_layer(variable_names: List[str], use_tf_style: bool) -> Dict[str, str]:
    """
    Collect recurrent variables by name. This only supports single-layer RNNs.
    """

    def get_key(name: str, gate: str) -> str:
        if '_w_' in name:
            return 'W_{0}'.format(gate)
        elif '_u_' in name:
            return 'U_{0}'.format(gate)
        return 'b_{0}'.format(gate)

    var_dict: Dict[str, str] = dict()
    if use_tf_style:
        for var_name in variable_names:
            name = var_name.lower()
            if 'candidate_kernel' in name:
                var_dict['candidates_kernel'] = var_name
            elif 'candidate_bias' in name:
                var_dict['candidates_bias'] = var_name
            elif 'gates_kernel' in name:
                var_dict['gates_kernel'] = var_name
            elif 'gates_bias' in name:
                var_dict['gates_bias'] = var_name
    else:
        for var_name in variable_names:
            name = var_name.lower()

            # Only get parameters from RNN cells
            if 'rnn_cell' not in name:
                continue

            if 'update' in name:
                var_dict[get_key(name, 'update')] = var_name
            elif 'reset' in name:
                var_dict[get_key(name, 'reset')] = var_name
            else:
                var_dict[get_key(name, 'candidate')] = var_name

    return var_dict


def to_fixed_point(value: float, num_bits: int) -> int:
    """
    Convert the given value to a fixed point representation with the number of bits.
    To handle the sign, we add an extra bit to mark the sign.
    """
    return int(value * pow(2, num_bits))


def quantize(weights: np.ndarray, num_bits: int) -> np.ndarray:
    """
    Quantize the given weights array into fixed point numbers with the given number of bits.
    """
    quantization_func = np.vectorize(partial(to_fixed_point, num_bits=num_bits)) 
    rounded_weights = quantization_func(weights)
    return rounded_weights


def compress_weights(model_parameters: Dict[str, np.ndarray], var_frac: float, num_bits: int) -> DefaultDict[str, List[np.ndarray]]:
    size_before = 0
    size_after = 0

    compressed: DefaultDict[str, List[np.ndarray]] = defaultdict(list)

    for weight_name, weight_mat in model_parameters.items():
        size_before += np.prod(weight_mat.shape)

        if len(weight_mat.shape) == 2 and weight_mat.shape[0] != 1 and abs(var_frac - 1.0) > SMALL_NUMBER:
            U, V = low_rank_decomposition(weight_mat, var_frac=var_frac)
            size_after += np.prod(U.shape) + np.prod(V.shape)

            compressed[weight_name].append(quantize(U, num_bits=num_bits))
            compressed[weight_name].append(quantize(V, num_bits=num_bits))
        else:
            size_after += np.prod(weight_mat.shape)
            compressed[weight_name].append(quantize(weight_mat, num_bits=num_bits))

    print('Size Before: {0}, Size After: {1}. Ratio: {2}'.format(size_before, size_after, size_after / size_before))
    return compressed


def get_name_prefix(weight_name: str) -> str:
    if 'embedding' in weight_name:
        weight_prefix = 'EMBEDDING'
    elif 'output' in weight_name:
        weight_prefix = 'OUTPUT'
    elif 'aggregate' in weight_name:
        weight_prefix = 'AGGREGATE'
    elif 'combine-states' in weight_name:
        weight_prefix = FUSION
    else:
        weight_prefix = 'TRANSFORM'
    
    return weight_prefix


def get_output_name(weight_name: str) -> str:
    dense_name, index = get_dense_name(weight_name)
    if 'hidden' in weight_name:
        dense_name = 'HIDDEN_{0}'.format(dense_name)
    return dense_name, index


def get_dense_name(weight_name: str) -> Tuple[str, int]:
    match = LAYER_REGEX.match(weight_name)
    index = int(match.group(1)) if match is not None else 0

    if 'bias' in weight_name.lower():
        return BIAS, index
    return KERNEL, index


def get_transform_name(weight_name: str) -> Tuple[str, int]:
    if 'rnn-cell' in weight_name:
        match = RNN_WEIGHT_REGEX.match(weight_name)

        var_name = match.group(1).upper() + '_' + match.group(3).upper()
        index = int(match.group(2))

        return var_name.replace('-', '_'), int(match.group(2))
    else:
        match = MULTI_RNN_CELL_REGEX.match(weight_name)
        if match is not None:
            # This is a tensorflow RNN Cell
            var_name = match.group(1).upper() + '_' + match.group(3).upper() + '_' + match.group(4).upper()
            index = int(match.group(2))
            var_name = '{0}_{1}'.format(var_name, index)

            return var_name.replace('-', '_'), index
        else:
            return get_dense_name(weight_name)


def convert_network(model_path: str, model_parameters: Dict[str, np.ndarray], num_bits: int, thresholds: Optional[List[float]], level: Optional[int]):
    # Dictionary to hold all variables
    weight_variables: List[str] = []
    dtype = 'dtype'  # The true data type is set via a typedef in the C implementation

    weight_layers: DefaultDict[str, int] = defaultdict(int)
    var_name_dict: Dict[str, str] = dict()  # Track variable renaming

    # Store the embedding layer variables
    for weight_name, weight_value in sorted(model_parameters.items()):

        if args.level is not None:
            match = LEVEL_REGEX.match(weight_name)

            weight_level = int(match.group(1))
            if weight_level != args.level - 1:
                continue

        # We transpose all 2d weights to standardize matrix and vector formats.
        # Tensorflow uses the transpose version of weight matrices.
        if len(weight_value.shape) == 2:
            weight_value = weight_value.T

        # (1) Extract the weight prefix
        name_prefix = get_name_prefix(weight_name)

        # (2) Extract the specific name
        var_name = ''
        layer_index = 0
        if name_prefix == 'TRANSFORM':
            var_name, index = get_transform_name(weight_name)
        elif name_prefix == 'OUTPUT':
            var_name, index = get_output_name(weight_name)
        else:
            var_name, index = get_dense_name(weight_name)

        if weight_name.endswith('-first'):
            var_name += '_FIRST'
        elif weight_name.endswith('-second'):
            var_name += '_SECOND'

        weight_layers[name_prefix + '_' + var_name] += 1

        # Quantize the weights and convert to fixed point numbers
        fixed_point_weights = [to_fixed_point(x, num_bits=num_bits) for x in weight_value.reshape(-1)]

        # (3) Save variables and associated data as C static variables
        n_dims = len(weight_value.shape)
        dims = weight_value.shape
        dims_str = seq_to_str(dims)
        flattened_weights = seq_to_str(fixed_point_weights)

        c_name = NAME_FORMAT.format(name_prefix, var_name, index)
        var_name_dict[weight_name] = '{0}_{1}'.format(c_name, MATRIX)

        n_dims_var = C_VAR_FORMAT.format('uint8_t', c_name + '_NUM_DIMS', n_dims)
        dims_var = C_VAR_FORMAT.format('uint8_t', ARRAY_FORMAT.format(c_name + '_DIMS', n_dims), dims_str)
        weights_var = C_VAR_FORMAT.format(dtype, ARRAY_FORMAT.format(c_name, np.prod(weight_value.shape)), flattened_weights)

        # We store everything as 2D matrices for consistency purposes
        if len(weight_value.shape) == 1:
            dim0 = weight_value.shape[0]
            dim1 = 1
        else:
            dim0 = weight_value.shape[0]
            dim1 = weight_value.shape[1]

        struct_code = create_matrix(c_name, dim0=dim0, dim1=dim1)
        weight_variables.extend([weights_var, struct_code + '\n'])

    # Store all weights associated with a particular layer into a single void * array
    layer_names: List[str] = []
    for weight_layer, count in weight_layers.items():
        layer_weight_names = ['{0}_{1}'.format(weight_layer, index) for index in range(count)]

        layer_weight_var = LAYER_ARRAY_FORMAT.format('{0}_LAYERS'.format(weight_layer), count, ', '.join(layer_weight_names))
        num_layers = C_VAR_FORMAT.format('int8_t', '{0}_NUM_LAYERS'.format(weight_layer), count)

        layer_names.extend([layer_weight_var, num_layers])

    # Get the model hyper-parameters
    hypers = get_hypers(model_path)
    model_class = hypers.model
    model_type = hypers.model_params['model_type']
    output_type = hypers.model_params['output_type']
    state_size = hypers.model_params['state_size']
    output_hidden_units = hypers.model_params['output_hidden_units']
    transform_type = hypers.model_params.get('rnn_cell_type', 'dense')
    is_compressed = 1 if hypers.model_params.get('compression_fraction', 1.0) < 1.0 else 0

    if level is None:
        seq_length = int(hypers.seq_length)
        samples_per_seq = int(seq_length * hypers.model_params.get('sample_frac', 1.0))
        num_sequences = int(seq_length / samples_per_seq)
    else:
        samples_per_seq = int(hypers.seq_length * hypers.model_params.get('sample_frac', 1.0))
        seq_length = level * samples_per_seq
        num_sequences = level

    # Get the input and output scalers
    metadata = get_metadata(model_path)
    input_shape = metadata['input_shape']
    assert len(input_shape) == 1, 'Can only support a single input dimension'

    num_input_features = input_shape[0]
    num_output_features = metadata['num_output_features'] if output_type != 'multi_classification' else metadata['num_classes']

    input_mean = metadata['input_scaler'].mean_
    input_std = metadata['input_scaler'].scale_

    if metadata.get('output_scaler') is not None:
        output_mean = metadata['output_scaler'].mean_
        output_std = metdata['output_scaler'].scale_
    else:
        output_mean = [0.0 for _ in range(num_output_features)]
        output_std = [1.0 for _ in range(num_output_features)]

    # Convert the scaling values to fixed point representation
    input_mean = [to_fixed_point(x, num_bits=num_bits) for x in input_mean]
    input_std = [to_fixed_point(x, num_bits=num_bits) for x in input_std]
    output_mean = [to_fixed_point(x, num_bits=num_bits) for x in output_mean]
    output_std = [to_fixed_point(x, num_bits=num_bits) for x in output_std]

    # Collect each layer into a dictionary to help with the graph creation
    var_names = list(var_name_dict.values())
    embedding_layer = collect_dense_layer(var_names, layer_name='embedding', activation=hypers.model_params['embedding_layer_params']['dense_activation'], activate_final=True)
    output_layer = collect_dense_layer(var_names, layer_name='output', activation=hypers.model_params['output_hidden_activation'], activate_final=True)

    # TODO: Add support for bag-of-words layers and infer the RNN style
    fusion_layer = None
    if (model_class == 'standard' and model_type == 'rnn'):
        rnn_layer = collect_recurrent_layer(var_names, use_tf_style=True)
    elif (model_class == 'adaptive' and model_type != 'bow'):
        rnn_layer = collect_recurrent_layer(var_names, use_tf_style=False)
        fusion_layer = collect_dense_layer(var_names, layer_name='fusion', activation='sigmoid', activate_final=True)

    # Convert the output thresholds to fixed point numbers
    if thresholds is None:
        thresholds = [1.0 for _ in range(num_sequences)]

    thresholds = [to_fixed_point(x, num_bits) for x in thresholds]

    model_parameters = {
        'embedding': embedding_layer,
        'transform': rnn_layer,
        'fusion': fusion_layer,
        'output': output_layer,
        'model_class': model_class,
        'model_type': model_type,
        'output_type': output_type,
        'output_hidden_units': output_hidden_units,
        'output_units': num_output_features,
        'state_size': state_size,
        'seq_length': seq_length,
        'transform_type': transform_type,
        'samples_per_seq': samples_per_seq,
        'is_compressed': is_compressed
    }
    save_by_file_suffix(model_parameters, 'model_parameters.pkl.gz')

    with open('neural_network.h', 'w') as output_file:
        output_file.write('#include <stdint.h>\n')
        output_file.write('#include "math/matrix.h"\n')
        output_file.write('#include "layers/cells.h"\n')
        output_file.write('#include "layers/layers.h"\n')
        output_file.write('#include "utils/neural_network_utils.h"\n')
        output_file.write('#include "utils/utils.h"\n')
        output_file.write('#include "math/matrix_ops.h"\n')
        output_file.write('#include "math/fixed_point_ops.h"\n\n')

        # C header guard
        output_file.write('#ifndef NEURAL_NETWORK_GUARD\n')
        output_file.write('#define NEURAL_NETWORK_GUARD\n\n')

        # Write the number of bits used during fixed point quantization
        output_file.write('#define FIXED_POINT_PRECISION {0}\n\n'.format(num_bits))
        
        # Write whether or not the dense layers are compressed
        output_file.write('#define IS_COMPRESSED {0}\n\n'.format(is_compressed))

        # Define ENUMs for the model and output types
        output_file.write('enum ModelClass { STANDARD = 0, ADAPTIVE = 1 };\n')
        output_file.write('enum ModelType { VANILLA = 0, SAMPLE = 1, BOW = 2, RNN = 3, BIRNN = 4 };\n')
        output_file.write('enum OutputType { REGRESSION = 0, BINARY_CLASSIFICATION = 1, MULTI_CLASSIFICATION = 2 };\n')

        # Write hyperparameters
        output_file.write(C_VAR_FORMAT.format('enum ModelClass', 'MODEL_CLASS', model_class.upper()) + '\n')
        output_file.write(C_VAR_FORMAT.format('enum ModelType', 'MODEL_TYPE', model_type.upper()) + '\n')
        output_file.write(C_VAR_FORMAT.format('enum OutputType', 'OUTPUT_TYPE', output_type.upper()) + '\n')
        output_file.write('#define {0} {1}\n'.format('STATE_SIZE', state_size))
        output_file.write('#define {0} {1}\n'.format('SEQ_LENGTH', seq_length))
        output_file.write('#define {0} {1}\n'.format('SAMPLES_PER_SEQ', samples_per_seq))
        output_file.write('#define {0} {1}\n'.format('NUM_SEQUENCES', num_sequences))
        output_file.write('#define {0} {1}\n'.format('NUM_INPUT_FEATURES', num_input_features))
        output_file.write('#define {0} {1}\n'.format('NUM_OUTPUT_FEATURES', num_output_features))
        output_file.write(C_VAR_FORMAT.format('int16_t', 'INPUT_MEAN[{0}]'.format(num_input_features), seq_to_str(input_mean)) + '\n')
        output_file.write(C_VAR_FORMAT.format('int16_t', 'INPUT_STD[{0}]'.format(num_input_features), seq_to_str(input_std)) + '\n')
        output_file.write(C_VAR_FORMAT.format('int16_t', 'OUTPUT_MEAN[{0}]'.format(num_output_features), seq_to_str(output_mean)) + '\n')
        output_file.write(C_VAR_FORMAT.format('int16_t', 'OUTPUT_STD[{0}]'.format(num_output_features), seq_to_str(output_std)) + '\n\n')
        output_file.write(C_VAR_FORMAT.format('int16_t', 'THRESHOLDS[{0}]'.format(num_sequences), seq_to_str(thresholds)) + '\n\n')

        for weight_var in weight_variables:
            output_file.write(weight_var)
            output_file.write('\n')

        output_file.write('\n')

        # Write the function prototype
        output_file.write('InferenceResult *execute_model(matrix *inputs[SEQ_LENGTH], InferenceResult *result, int16_t num_sequences);\n')

        output_file.write('#endif\n')


if __name__ == '__main__':
    parser = ArgumentParser('Compresses the neural network and converts the parameters into a C header file.')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--precision', type=int, required=True)
    parser.add_argument('--optimized-test-log', type=str)
    parser.add_argument('--level', type=int, help='Level to keep (1-indexed)')
    args = parser.parse_args()

    assert args.precision > 0 and args.precision < 16, 'The precision must be in [1, 15]'

    model_parameters = read_by_file_suffix(args.model_path)

    thresholds = None
    if args.optimized_test_log is not None:
        opt_test_log = list(read_by_file_suffix(args.optimized_test_log))[0]
        thresholds = opt_test_log['THRESHOLDS']

    convert_network(args.model_path, model_parameters, num_bits=args.precision, thresholds=thresholds, level=args.level)
