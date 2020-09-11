import re
import numpy as np
from functools import partial
from typing import List, Union, Optional, Any, Tuple


# Constants used during variable naming
BIAS = 'BIAS'
KERNEL = 'KERNEL'
MATRIX = 'MAT'
HIDDEN = 'HIDDEN'
ACTIVATION = 'ACTIVATION'

# Constants used to format results as C variables
STATIC_VAR_FORMAT = 'static {0} {1} = {2};'

# Regular Expression to parse a variable name
VARIABLE_REGEX = re.compile(r'model\/([^_-]+[_-][^_-]+)-([^:]+):0')


def should_use_lea_ram(mat: np.ndarray) -> bool:
    return mat.shape[0] > 1 and mat.shape[1] > 1


def parse_variable_name(variable_name: str) -> Tuple[str, str]:
    match = VARIABLE_REGEX.match(variable_name)
    assert match is not None, 'Could not parse {0}'.format(variable_name)
    return match.group(1), match.group(2)


def array_to_string(array: Union[List[Any], np.ndarray]) -> str:
    """
    Formats the 1d array as a comma-separated string enclosed in braces.
    """
    # Validate shapes
    if isinstance(array, np.ndarray):
        assert len(array.shape) == 1, 'Can only format 1d arrays'

    return '{{ {0} }}'.format(','.join(map(str, array)))


def float_to_fixed_point(x: float, precision: int) -> int:
    """
    Converts the given floating point value to fixed point representation
    with the given number of fractional bits.
    """
    multiplier = 1 << precision

    width = 16 if precision >= 8 else 8
    max_val = (1 << (width - 1)) - 1
    min_val = -max_val

    fp_val = int(round(x * multiplier))

    if fp_val > max_val:
        print('WARNING: Observed positive overflow')
        return max_val
    elif fp_val < min_val:
        print('WARNING: Observed negative overflow')
        return min_val
    return fp_val


def tensor_to_fixed_point(tensor: Union[List[float], np.ndarray], precision: int) -> Union[List[int], np.ndarray]:
    """
    Converts each element in the given tensor to fixed point representation with the given
    number of fractional bits.
    """
    fixed_point_converter = partial(float_to_fixed_point, precision=precision)

    if isinstance(tensor, np.ndarray):
        fp_function = np.vectorize(fixed_point_converter)
        return fp_function(tensor)
    else:
        return list(map(fixed_point_converter, tensor))


def create_constant(name: str, value: Optional[int], should_add_newline: bool = True) -> str:
    if value is None:
        const = '#define {0}'.format(name)
    else:
        const = '#define {0} {1}'.format(name, value)

    if should_add_newline:
        return const + '\n'
    return const


def create_static_variable(name: str, dtype: str, value: str) -> str:
    return STATIC_VAR_FORMAT.format(dtype, name, value)


def create_matrix(name: str, mat: np.ndarray, is_msp: bool) -> str:
    """
    Creates a variables associated with a weight matrix. Each matrix
    is composed of three variables:
        (1) {NAME}: A 2d array containing the weights
        (2) {NAME}_MATRIX_VAR: A matrix struct for this weight matrix
        (3) {NAME}_MATRIX: A pointer to the {NAME}_MATRIX_VAR variable
    We separate these by newlines and return the definitions in a single string.

    Args:
        name: Name of this trainable parameter
        mat: A 1d or 2d array containing the model parameters. For consistency,
            all 1d arrays are given a column size of 1 and reshaped to 2d arrays.
            The number of rows must be even for compatibility with the Low-Energy Accelerator.
        is_msp: Whether to define the variables for the MSP device. This entails specifying the
            memory location for the data matrix.
    Returns:
        The four variable definitions in a single newline-separated string.
    """
    assert len(mat.shape) == 1 or len(mat.shape) == 2, 'Can only create matrices of at most 2 dimensions'
    assert mat.shape[0] % 2 == 0 or mat.shape[0] == 1, 'The number of rows must be even or larger than 1. Got: {0}'.format(mat.shape)

    # Ensure the matrix is a 2d array and unpack the dimensions
    if len(mat.shape) == 1:
        mat = np.expand_dims(mat, axis=-1)  # [D, 1]

    dim0, dim1 = mat.shape

    declarations: List[str] = []

    # 1) Create the weight matrix array. We package everything into 1d arrays.
    # As a note, we only place matrices with dimensions > 1 into LEA RAM. The
    # operations for these matrices are done without the LEA to avoid overhead.
    if is_msp and should_use_lea_ram(mat):
        fram_pragma = 'DSPLIB_DATA({0}, 4)'.format(name)
        declarations.append(fram_pragma)

    matrix_string = array_to_string(mat.reshape(-1))
    weight_mat_variable = create_static_variable(name='{0}[{1}]'.format(name, dim0 * dim1),
                                                 dtype='dtype',
                                                 value=matrix_string)
    declarations.append(weight_mat_variable)

    # 2) Create the matrix struct. We initialze this variable through unpacking.
    matrix_variable_name = '{0}_{1}_VAR'.format(name, MATRIX)
    matrix_variable = create_static_variable(name=matrix_variable_name,
                                             dtype='matrix',
                                             value='{{ {0}, {1}, {2} }}'.format(name, dim0, dim1))
    declarations.append(matrix_variable)

    # 3) Create the matrix pointer
    ptr_variable = create_static_variable(name='{0}_{1}'.format(name, MATRIX),
                                          dtype='matrix *',
                                          value='&{0}'.format(matrix_variable_name))
    declarations.append(ptr_variable)

    return '\n'.join(declarations)


def create_array(array: Union[List[float], np.ndarray], name: str, dtype: str) -> str:
    """
    Creates a C variable for the given array.

    Args:
        array: A 1d or 2d array to extract into a C variable
        name: Name of the C variable
        dtype: Data type of the resulting C variable
    Returns:
        The C declaration and initialization of the array
    """
    if isinstance(array, np.ndarray) and len(array.shape) == 2:
        rows: List[str] = []
        for row in array:
            rows.append(array_to_string(row))

        array_string = '{{ {0} }}'.format(','.join(rows))

        dim0, dim1 = array.shape
        array_variable = '{0}[{1}][{2}]'.format(name, dim0, dim1)
    else:
        array_string = array_to_string(array)
        array_variable = '{0}[{1}]'.format(name, len(array))

    return create_static_variable(name=array_variable,
                                  dtype=dtype,
                                  value=array_string)
