#include "matrix_ops.h"


matrix *matrix_allocate(int8_t numRows, int8_t numCols) {
    matrix *mat = (matrix *) alloc(sizeof(matrix));
    if (isNull(mat)) {
        return NULL_PTR;
    }

    int16_t *data = (int16_t *) alloc(numRows * numCols * sizeof(int16_t));
    if (isNull(data)) {
        free(mat);
        return NULL_PTR;
    }

    mat->numRows = numRows;
    mat->numCols = numCols;
    mat->data = data;
    return mat;
}


matrix *matrix_create_from(int16_t *data, int8_t numRows, int8_t numCols) {
    matrix *mat = (matrix *) alloc(sizeof(matrix));
    if (isNull(mat)) {
        return NULL_PTR;
    }

    mat->numRows = numRows;
    mat->numCols = numCols;
    mat->data = data;
    return mat;
}


void matrix_free(matrix *mat) {
    if (isNull(mat)) {
        return;
    }

    if (!isNull(mat->data)) {
        free(mat->data);
    }

    free(mat);
}


matrix *matrix_add(matrix *result, matrix *mat1, matrix *mat2) {
    /**
     * Adds two matrices together elementwise.
     * Result stored in matrix 1.
     */
    
    // Validate dimensions
    if ((mat1->numRows != mat2->numRows) || (mat1->numCols != mat2->numCols) ||
        (result->numRows != mat1->numRows) || (result->numCols != mat2->numCols)) {
        return NULL_PTR;
    }
    
    // Compute elementwise sum in place
    for (int16_t i = 0; i < mat1->numRows * mat1->numCols; i++) {
        result->data[i] = fp_add(mat1->data[i], mat2->data[i]);
    }

    return result;
}


matrix *matrix_multiply(matrix *result, matrix *mat1, matrix *mat2, int16_t precision) {
    /**
     * Performs matrix multiplication and stores value in given result array
     */

    // Validate dimensions
    if ((mat1->numCols != mat2->numRows) || (mat1->numRows != result->numRows) || (mat2->numCols != result->numCols)) {
        return NULL_PTR;
    }

    // The result will be a [n, p] matrix
    int16_t n = mat1->numRows;
    int16_t m = mat1->numCols;
    int16_t p = mat2->numCols;

    for (int16_t i = 0; i < n; i++) {
        int16_t outerRow = i * m;  // Offset for the i^th row

        for (int16_t j = 0; j < p; j++) {
            int16_t sum = 0;

            for (int16_t k = 0; k < m; k++) {
                int16_t innerRow = k * p;  // Offset for the k^th row
                
                int16_t prod = fp_mul(mat1->data[outerRow + k], mat2->data[innerRow + j], precision);
                sum = fp_add(sum, prod);
            }
     
            int16_t resultRow = i * p;
            result->data[resultRow + j] = sum;
        }
    }

    return result;
}


matrix *matrix_hadamard(matrix* result, matrix *mat1, matrix *mat2, int16_t precision) {
    /**
     * Elementwise matrix product. Result stored in matrix 1.
     */
    
    // Validate dimensions
    if ((mat1->numRows != mat2->numRows) || (mat1->numCols != mat2->numCols) ||
        (result->numRows != mat1->numRows) || (result->numCols != mat1->numCols)) {
        return NULL_PTR;
    }

    for (int16_t i = 0; i < mat1->numRows * mat1->numCols; i++) {
        result->data[i] = fp_mul(mat1->data[i], mat2->data[i], precision);
    }

    return result;
}


matrix *scalar_product(matrix *result, matrix *mat, int16_t scalar, int16_t precision) {
    /**
     * Multiplies every element in the matrix by the given scalar.
     * Result stored directly into the given array.
     */
    // Validate dimensions
    if ((result->numRows != mat->numRows) || (result->numCols != mat->numCols)) {
        return NULL_PTR;
    }

    for (int16_t i = 0; i < mat->numRows * mat->numCols; i++) {
        result->data[i] = fp_mul(mat->data[i], scalar, precision);
    }

    return result;
}


matrix *scalar_add(matrix *result, matrix *mat, int16_t scalar) {
    /**
     * Adds the given scalar to every element of the matrix.
     * Stores result directly in the matrix.
     */
    // Validate dimensions
    if ((result->numRows != mat->numRows) || (result->numCols != mat->numCols)) {
        return NULL_PTR;
    }

    for (int16_t i = 0; i < mat->numRows * mat->numCols; i++) {
        result->data[i] = fp_add(mat->data[i], scalar);
    }
    
    return result;
}


matrix *apply_elementwise(matrix *result, matrix *mat, int16_t (*fn)(int16_t, int16_t), int16_t precision) {
    /**
     * Applies the given function to every element of the
     * input matrix. Result stored directly in the matrix.
     */
    // Validate dimensions
    if ((result->numRows != mat->numRows) || (result->numCols != mat->numCols)) {
        return NULL_PTR;
    }

    for (int16_t i = 0; i < mat->numRows * mat->numCols; i++) {
        result->data[i] = (*fn)(mat->data[i], precision);
    }

    return result;
}


matrix *transpose(matrix *result, matrix *mat) {
    /**
     * Transposes the given matrix and stores the value
     * in the given result matrix.
     */
    if (mat->numCols != result->numRows || mat->numRows != result->numCols) {
        return NULL_PTR;
    }

    int16_t n = mat->numRows;
    int16_t m = mat->numCols;

    for (int16_t i = 0; i < n; i++) {
        for (int16_t j = 0; j < m; j++) {
            result->data[j * n + i] = mat->data[i * m + j];
        }
    }

    return result;
}


matrix *matrix_replace(matrix *dst, matrix *src) {
    /**
     * Replaces the contents of the destination matrix with those from the src.
     */
    if ((dst->numRows != src->numRows) || (dst->numCols != src->numCols)) {
        return NULL_PTR;
    }
    
    for (int16_t i = 0; i < dst->numRows * dst->numCols; i++) {
        dst->data[i] = src->data[i];
    }

    return dst;
}


matrix *matrix_set(matrix *mat, int16_t value) {
    /**
     * Sets all values in the matrix to the given value (already in fixed point form).
     */
    for (int16_t i = 0; i < mat->numRows * mat->numCols; i++) {
        mat->data[i] = value;
    }

    return mat;
}


matrix *stack(matrix *result, matrix *vec1, matrix *vec2) {
    /**
     * Stacks the two vectors into a larger vector.
     */
    // Validate the input shapes.
    if ((vec1->numRows + vec2->numRows != result->numRows) || (vec1->numCols != 1) ||
        (vec2->numCols != 1) || (result->numCols != 1)) {
        return NULL_PTR;
    }

    int16_t index = 0;
    for (; index < vec1->numRows; index++) {
        result->data[index] = vec1->data[index];
    }

    int16_t offset = index;
    for (; index < result->numRows; index++) {
        result->data[index] = vec2->data[index - offset];
    }

    return result;
}


int16_t argmax(matrix *vec) {
    /**
     * Computes the argmax of the 1d vector.
     */
    if (vec->numCols != 1 || vec->numRows <= 0) {
        return -1;
    }

    int16_t max = vec->data[0];
    int16_t max_index = 0;
    for (int16_t i = 1; i < vec->numRows; i++) {
        if (vec->data[i] > max) {
            max_index = i;
            max = vec->data[i];
        }
    }

    return max_index;
}


matrix *normalize(matrix *vec, int16_t *mean, int16_t *std, int16_t precision) {
    /**
     * Normalizes the vector to a standard normal distribution. This operation is in-place,
     * so the original vector is mutated.
     */
    if (vec->numCols != 1) {
        return NULL_PTR;
    }

    int16_t shifted;
    for (int16_t i = 0; i < vec->numRows; i++) {
        shifted = fp_sub(vec->data[i], mean[i]);
        vec->data[i] = fp_div(shifted, std[i], precision);
    }

    return vec;
}
