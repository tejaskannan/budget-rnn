import numpy as np
import re
import os.path
from argparse import ArgumentParser
from collections import defaultdict
from typing import Callable, Set, Tuple, Dict, List, DefaultDict, Any
from functools import partial

from utils.hyperparameters import HyperParameters
from utils.file_utils import read_by_file_suffix, extract_model_name
from utils.constants import SMALL_NUMBER, BIG_NUMBER, HYPERS_PATH, METADATA_PATH


BIAS = 'BIAS'
KERNEL = 'KERNEL'
LAYER_REGEX = re.compile('.*/.*-([0-9]+).*')
RNN_WEIGHT_REGEX = re.compile('.*/(rnn-cell)-cell-([0-9+])-(.*):.*')

NAME_FORMAT = '{0}_{1}_{2}'
LAYER_ARRAY_FORMAT = 'static void * {0}[{1}] = {{ {2} }};'
OUTPUT_NAME_REGEX = re.compile('.*/output-.*(hidden)')
C_VAR_FORMAT = 'static {0} {1} = {2};'
ARRAY_FORMAT = '{0}[{1}]'


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


def low_rank_decomposition(mat: np.ndarray, var_frac: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decomposes the given matrix into two low-rank matrices using the singular value decomposition.

    Args:
        mat: A [N, M] matrix
        rank: Desired rank (R) of the output matrix
    Returns:
        A pair of matrices. The first is an [N, R] matrix and the second is a [R, M]
        matrix. The product of these matrices is the low-rank approximation.
    """
    if abs(var_frac - 1.0) < SMALL_NUMBER:
        return mat

    U, S, VH = np.linalg.svd(mat, compute_uv=True)

    eigenvalues = np.square(S)
    eigenvalue_sum = np.sum(eigenvalues)
    
    variance_fractions = np.cumsum(eigenvalues) / eigenvalue_sum
    rank = np.sum((variance_fractions <= var_frac).astype(int))

    # Pad singular values with zeros to align the shapes
    S = np.pad(S, pad_width=(0, U.shape[1] - S.shape[0]), mode='constant', constant_values=0)
    singular_values = np.diag(S)
    US = U.dot(singular_values)
    
    U_low_rank = US[:, :rank]
    V_low_rank = VH[:rank, :]

    return U_low_rank, V_low_rank


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

        if len(weight_mat.shape) == 2 and weight_mat.shape[0] != 1:
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
    return get_dense_name(weight_name)


def compress_network(model_path: str, model_parameters: Dict[str, np.ndarray], var_frac: float, num_bits: int):
    compressed_parameters = compress_weights(model_parameters, var_frac, num_bits)

    # Rename weights in the compressed parameters dictionary
    compressed_model_params: Dict[str, np.ndarray] = dict()
    for name, weights in compressed_parameters.items():
        assert len(weights) == 1 or len(weights) == 2, 'Should only have 1 or 2 weight components'

        if len(weights) == 2:
            first_name = '{0}-first'.format(name)
            second_name = '{0}-second'.format(name)

            compressed_model_params[first_name] = weights[0]
            compressed_model_params[second_name] = weights[1]
        else:
            compressed_model_params[name] = weights[0]

    # Dictionary to hold all variables
    weight_variables: List[str] = []
    dtype = 'int8_t' if num_bits <= 8 else 'int16_t' 

    weight_layers: DefaultDict[str, int] = defaultdict(int)

    # Store the embedding layer variables
    for weight_name, weight_value in sorted(compressed_model_params.items()):

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

        # (3) Save variables and associated data as C static variables
        n_dims = len(weight_value.shape)
        dims = weight_value.shape
        dims_str = seq_to_str(dims)
        flattened_weights = seq_to_str(weight_value.reshape(-1))

        c_name = NAME_FORMAT.format(name_prefix, var_name, index)

        n_dims_var = C_VAR_FORMAT.format('int8_t', c_name + '_NUM_DIMS', n_dims)
        dims_var = C_VAR_FORMAT.format('int8_t', ARRAY_FORMAT.format(c_name + '_DIMS', n_dims), dims_str)
        weights_var = C_VAR_FORMAT.format('fixed_point', ARRAY_FORMAT.format(c_name, np.prod(weight_value.shape)), flattened_weights)

        weight_variables.extend([n_dims_var, dims_var, weights_var])

    # Store all weights associated with a particular layer into a single void * array
    layer_names: List[str] = []
    for weight_layer, count in weight_layers.items():
        layer_weight_names = ['{0}_{1}'.format(weight_layer, index) for index in range(count)]

        layer_weight_var = LAYER_ARRAY_FORMAT.format('{0}_LAYERS'.format(weight_layer), count, ', '.join(layer_weight_names))
        num_layers = C_VAR_FORMAT.format('int8_t', '{0}_NUM_LAYERS'.format(weight_layer), count)

        layer_names.extend([layer_weight_var, num_layers])

    # Get the model type and output type
    hypers = get_hypers(model_path)
    model_class = hypers.model
    model_type = hypers.model_params['model_type']
    output_type = hypers.model_params['output_type']
    state_size = hypers.model_params['state_size']
    transform_type = hypers.model_params.get('rnn_cell_type', 'dense')
    seq_length = hypers.seq_length
    samples_per_seq = seq_length * hypers.model_params.get('sample_frac', 1.0)

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

    with open('neural_network.h', 'w') as output_file:
        output_file.write('#include <stdint.h>\n\n')

        output_file.write('typedef {0} fixed_point;\n\n'.format(dtype))

        # C header guard
        output_file.write('#ifndef NEURAL_NETWORK_GUARD\n')
        output_file.write('#define NEURAL_NETWORK_GUARD\n\n')

        # Write the number of bits used during fixed point quantization
        output_file.write(C_VAR_FORMAT.format('int', 'NUM_FIXED_POINT_BITS', num_bits) + '\n\n')

        # Define ENUMs for the model and output types
        output_file.write('enum ModelClass { STANDARD = 0, ADAPTIVE = 1 };\n')
        output_file.write('enum ModelType { VANILLA = 0, SAMPLE = 1, BOW = 2 };\n')
        output_file.write('enum OutputType { REGRESSION = 0, BINARY_CLASSIFICATION = 1, MULTI_CLASSIFICATION = 2 };\n')
        output_file.write('enum TransformType { DENSE = 0, GRU = 1 };\n\n')

        # Write hyperparameters
        output_file.write(C_VAR_FORMAT.format('enum ModelClass', 'MODEL_CLASS', model_class.upper()) + '\n')
        output_file.write(C_VAR_FORMAT.format('enum ModelType', 'MODEL_TYPE', model_type.upper()) + '\n')
        output_file.write(C_VAR_FORMAT.format('enum OutputType', 'OUTPUT_TYPE', output_type.upper()) + '\n')
        output_file.write(C_VAR_FORMAT.format('enum TransformType', 'TRANSFORM_TYPE', transform_type.upper()) + '\n')
        output_file.write(C_VAR_FORMAT.format('int', 'STATE_SIZE', state_size) + '\n')
        output_file.write(C_VAR_FORMAT.format('int', 'SEQ_LENGTH', seq_length) + '\n')
        output_file.write(C_VAR_FORMAT.format('int', 'SAMPLES_PER_SEQ', samples_per_seq) + '\n')
        output_file.write(C_VAR_FORMAT.format('int', 'NUM_INPUT_FEATURES', num_input_features) + '\n')
        output_file.write(C_VAR_FORMAT.format('int', 'NUM_OUTPUT_FEATURES', num_output_features) + '\n')
        output_file.write(C_VAR_FORMAT.format('int', 'INPUT_MEAN[{0}]'.format(num_input_features), seq_to_str(input_mean)) + '\n')
        output_file.write(C_VAR_FORMAT.format('int', 'INPUT_STD[{0}]'.format(num_input_features), seq_to_str(input_std)) + '\n')
        output_file.write(C_VAR_FORMAT.format('int', 'OUTPUT_MEAN[{0}]'.format(num_output_features), seq_to_str(output_mean)) + '\n')
        output_file.write(C_VAR_FORMAT.format('int', 'OUTPUT_STD[{0}]'.format(num_output_features), seq_to_str(output_std)) + '\n\n')

        for weight_var in weight_variables:
            output_file.write(weight_var)
            output_file.write('\n')
    
        output_file.write('\n')

        for layer_var in layer_names:
            output_file.write(layer_var)
            output_file.write('\n')

        output_file.write('#endif\n')


if __name__ == '__main__':
    parser = ArgumentParser('Compresses the neural network and converts the parameters into a C header file.')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--variance-frac', type=float, required=True)
    parser.add_argument('--quantize-level', type=str, choices=['byte', 'short'])
    args = parser.parse_args()

    assert args.variance_frac > 0 and args.variance_frac <= 1, 'The variance fraction must be in (0, 1]'

    model_parameters = read_by_file_suffix(args.model_path)
    num_bits = 6 if args.quantize_level == 'byte' else 14

    compress_network(args.model_path, model_parameters, num_bits=num_bits, var_frac=args.variance_frac)
