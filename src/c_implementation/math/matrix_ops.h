#include <stdint.h>
#include "matrix.h"
#include "../memory.h"
#include "fixed_point_ops.h"
#include "hashing.h"
#include "../utils/string_utils.h"

#ifndef MATRIX_OPS_GUARD
    #define MATRIX_OPS_GUARD
    
    matrix *matrix_allocate(int8_t n_rows, int8_t n_cols);
    matrix *matrix_create_from(dtype *data, int8_t n_rows, int8_t n_cols);
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
    matrix *stack(matrix *result, matrix *vec1, matrix *vec2);
    int16_t argmax(matrix *vec);
    matrix *normalize(matrix *vec, const int16_t *mean, const int16_t *std, int16_t precision);
    matrix *hashed_matrix_vector_product(matrix *result, matrix *mat, matrix *vec, char *seed, uint8_t should_transpose, int16_t precision);
    int8_t threshold_prediction(matrix *logits, int16_t threshold, int16_t precision, dtype *temp_buffer);
    int16_t matrix_sum(matrix *mat);
    int16_t matrix_min(matrix *mat);

#endif
