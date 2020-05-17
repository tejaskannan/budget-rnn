from typing import List


MATRIX = 'matrix'
MATMUL = 'matrix_multiply'
ADD = 'matrix_add'
TRANSPOSE = 'matrix_transpose'
ALLOCATE = 'matrix_alloc'
FREE = 'matrix_free'
APPLY = 'matrix_apply'
ARGMAX = 'argmax'
FIRST_DIM = 'dim0'
SECOND_DIM = 'dim1'


def create_definition(function_name: str, argument_names: List[str]) -> str:
    arguments = ['matrix {0}'.format(name) for name in argument_names]
    return 'void {0}({1}) {{'.format(function_name, ', '.join(arguments))


def create_matmul(first_mat: str, second_mat: str, output_mat: str, transpose_first: bool, transpose_second: bool) -> str:
    result_lines: List[str] = []

    # Allocate space for the new matrix
    result_lines.append('{0} = {1}({2}->{3}, {4}->{5});'.format(output_mat, ALLOCATE, first_mat, FIRST_DIM, second_mat, SECOND_DIM))

    arg1, arg2 = first_mat, second_mat

    if transpose_first:
        first_transpose_name = '{0}Transpose'.format(first_mat)

        # Allocate new matrix and compute the transpose
        result_lines.append('{0} = {1}({2}->{3}, {4}->{5});'.format(first_transpose_name, ALLLOCATE, first_mat, SECOND_DIM, first_mat, FIRST_DIM))
        result_lines.append('{0}({1}, {2});\n'.format(TRANSPOSE, first_mat, first_transpose_name))

        arg1 = first_transpose_name

    if transpose_second:
        second_transpose_name = '{0}Transpose'.format(second_mat)

        # Allocate new matrix and compute the transpose
        result_lines.append('{0} = {1}({2}->{3}, {4}->{5});'.format(second_transpose_name, ALLLOCATE, second_mat, SECOND_DIM, second_mat, FIRST_DIM))
        result_lines.append('{0}({1}, {2});'.format(TRANSPOSE, second_mat, second_transpose_name))

        arg2 = second_transpose_name

    # Write the matrix multiplication
    result_lines.append('{0}({1}, {2}, {3});'.format(MATMUL, arg1, arg2, output_mat))

    # Free any temporary matrices
    if transpose_first:
        result_lines.append('{0}({1});'.format(FREE, first_transpose_name));

    if transpose_second:
        result_lines.append('{0}({1});'.format(FREE, second_transpose_name));

    return '\n'.join(result_lines)


def create_add(first_mat: str, second_mat: str, output_mat: str) -> str:
    result_lines: List[str] = []
    result_lines.append('{0} = {1}({2}->{3}, {4}->{5});'.format(output_mat, ALLOCATE, first_mat, FIRST_DIM, first_mat, SECOND_DIM))
    result_lines.append('{0}({1}, {2}, {3});'.format(ADD, first_mat, second_mat, output_mat))
    return '\n'.join(result_lines)


def create_activation(mat: str, output_mat: str, activation: str) -> str:
    result_lines: List[str] = []
    result_lines.append('{0} = {1}({2}->{3}, {4}->{5});'.format(output_mat, ALLOCATE, mat, FIRST_DIM, mat, SECOND_DIM))
    result_lines.append('{0}({1}, {2}, &{3});'.format(APPLY, mat, output_mat, activation))
    return '\n'.join(result_lines)


def create_softmax_prediction(mat: str, output_mat: str) -> str:
    return '{0}({1}, {2});\n'.format(ARGMAX, mat, output_mat)
