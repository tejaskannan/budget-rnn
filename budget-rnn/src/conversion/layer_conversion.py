import numpy as np

from .conversion_utils import create_matrix, tensor_to_fixed_point, float_to_fixed_point, create_constant


def weight_matrix_conversion(layer_name: str, weight_name: str, weight_matrix: np.ndarray, precision: int, is_msp: bool) -> str:
    """
    Converts the given Tensorflow variable to a C variable that is quantized in the given fixed
    point precision.

    Args:
        layer_name: Name of the layer containing this variable
        weight_name: Name of the variable
        weight_matrix: The corresponding weight matrix. This can be a 1d or 2d tensor
        precision: The number of fractional bits used during fixed point quantization
        is_msp: Whether to define the variables for the MSP device. This controls whether to directly
            specify the memory location
    Returns:
        The C variables associated with this weight matrix. The C variables are newline separated.
    """
    assert len(weight_matrix.shape) <= 2, 'Weight matrix can be at most 2 dimensions'

    # For 2d matrices, we always transpose the weights. In Tensorflow, a standard dense layer uses the format
    # (x^T)(W). In the embedded implementation, we instead use (W^T)x. This is purely a convention--the embedded
    # implementation uses a row-based vector format.
    if len(weight_matrix.shape) == 2:
        weight_matrix = weight_matrix.T

    # Create the C variable name for this trainable parameter
    c_variable_name = '{0}_{1}'.format(layer_name.upper(), weight_name.upper())
    c_variable_name = c_variable_name.replace('-', '_')

    # Compress the weight matrix using fixed point values
    if len(weight_matrix.shape) == 0:
        quantized_value = float_to_fixed_point(weight_matrix, precision=precision)
        return create_constant(name=c_variable_name, value=quantized_value, should_add_newline=False)
    else:
        quantized_matrix = tensor_to_fixed_point(weight_matrix, precision=precision)
        return create_matrix(name=c_variable_name, mat=quantized_matrix, is_msp=is_msp)
