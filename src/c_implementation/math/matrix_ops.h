#include <stdint.h>
#include "matrix.h"
#include "../memory.h"
#include "fixed_point_ops.h"

#ifndef TENSOR_GUARD
    #define TENSOR_GUARD
    
    matrix *matrix_allocate(int8_t n_rows, int8_t n_cols);
    matrix *matrix_create_from(int16_t *data, int8_t n_rows, int8_t n_cols);
    void matrix_free(matrix *mat);
    matrix *matrix_add(matrix *result, matrix *mat1, matrix *mat2);
    matrix *matrix_multiply(matrix *result, matrix *mat1, matrix *mat2, int16_t precision);
    matrix *matrix_hadamard(matrix *result, matrix *mat1, matrix *mat2, int16_t precision);
    matrix *scalar_product(matrix *result, matrix *mat, int16_t scalar, int16_t precision);
    matrix *scalar_add(matrix *result, matrix *mat, int16_t scalar);
    matrix *apply_elementwise(matrix *result, matrix *mat, int16_t (*fn)(int16_t, int16_t), int16_t precision);
    matrix *matrix_set(matrix *mat, int16_t value);
    matrix *transpose(matrix *result, matrix *mat);
    matrix *matrix_replace(matrix *dst, matrix *src);

#endif
