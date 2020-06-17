#include "matrix_ops.h"

// Temporary buffer for hash strings
static char hash_str[100];


matrix *matrix_allocate(int8_t numRows, int8_t numCols) {
    matrix *mat = (matrix *) alloc(sizeof(matrix));
    if (isNull(mat)) {
        return NULL_PTR;
    }

    dtype *data = (dtype *) alloc(numRows * numCols * sizeof(dtype));
    if (isNull(data)) {
        dealloc(mat);
        return NULL_PTR;
    }

    mat->numRows = numRows;
    mat->numCols = numCols;
    mat->data = data;
    return mat;
}


matrix *matrix_create_from(dtype *data, int8_t numRows, int8_t numCols) {
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
        dealloc(mat->data);
    }

    dealloc(mat);
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
    uint16_t i;
    for (i = mat1->numRows * mat1->numCols; i > 0; i--) {
        result->data[i - 1] = fp_add(mat1->data[i - 1], mat2->data[i - 1]);
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
    int8_t n = mat1->numRows;
    int8_t m = mat1->numCols;
    int8_t p = mat2->numCols;

    uint16_t i, j, k;
    for (i = n; i > 0; i--) {
        uint16_t outerRow = (i - 1) * m;  // Offset for the i^th row

        for (j = p; j > 0; j--) {
            int16_t sum = 0;

            for (k = m; k > 0; k--) {
                uint16_t innerRow = (k - 1) * p;  // Offset for the k^th row
                
                int16_t prod = fp_mul(mat1->data[outerRow + (k - 1)], mat2->data[innerRow + (j - 1)], precision);
                sum = fp_add(sum, prod);
            }
     
            uint16_t resultRow = (i - 1) * p;
            result->data[resultRow + (j - 1)] = sum;
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

    uint16_t i;
    for (i = mat1->numRows * mat1->numCols; i > 0; i--) {
        result->data[i - 1] = fp_mul(mat1->data[i - 1], mat2->data[i - 1], precision);
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

    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        result->data[i - 1] = fp_mul(mat->data[i - 1], scalar, precision);
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

    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        result->data[i - 1] = fp_add(mat->data[i - 1], scalar);
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

    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        result->data[i - 1] = (*fn)(mat->data[i - 1], precision);
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

    int8_t n = mat->numRows;
    int8_t m = mat->numCols;

    uint16_t i, j;
    for (i = n; i > 0; i--) {
        for (j = m; j > 0; j--) {
            result->data[(j - 1) * n + (i - 1)] = mat->data[(i - 1) * m + (j - 1)];
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

    uint16_t i;
    for (i = dst->numRows * dst->numCols; i > 0; i--) {
        dst->data[i - 1] = src->data[i - 1];
    }

    return dst;
}


matrix *matrix_set(matrix *mat, int16_t value) {
    /**
     * Sets all values in the matrix to the given value (already in fixed point form).
     */

    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        mat->data[i - 1] = value;
    }

    return mat;
}


int16_t matrix_sum(matrix *mat) {
    /**
     * Computes the sum of all elements in the matrix
     */
    int16_t sum = 0;
    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        sum = fp_add(mat->data[i - 1], sum);
    }

    return sum;
}


int16_t matrix_min(matrix *mat) {
    /**
     * Computes the minimum value over all elements in the matrix
     */
    int16_t min_value = 32767;  // 2^15 - 1
    int16_t val;
    uint16_t i;
    for (i = mat->numRows * mat->numCols; i > 0; i--) {
        val = mat->data[i - 1];
        if (val < min_value) {
            min_value = val;
        }
    }

    return min_value;
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

    uint16_t i;
    int16_t val;
    for (i = vec->numRows; i > 1; i--) {
        val = vec->data[i - 1];
        if (val > max) {
            max_index = i - 1;
            max = val;
        }
    }

    return max_index;
}


int8_t threshold_prediction(matrix *logits, int16_t threshold, int16_t precision, dtype *temp_buffer) {
    /**
     * Returns the highest-probability class if the normalized log probability is above the given threshold.
     * If the threshold is not satisfied, then the function returns -1.
     */
    if (logits->numCols != 1) {
        return -1;
    }

    // Allocate a temp matrix
    matrix temp;
    temp.numRows = logits->numRows;
    temp.numCols = logits->numCols;
    temp.data = temp_buffer;
    matrix *tempMat = &temp;

    tempMat = matrix_replace(tempMat, logits);

    int16_t min_logit = matrix_min(logits);
    tempMat = scalar_add(tempMat, tempMat, fp_neg(min_logit));
    
    int16_t logit_sum = matrix_sum(tempMat);
    int16_t normalize_factor = fp_div(int_to_fp(1, precision), logit_sum, precision);
    tempMat = scalar_product(tempMat, tempMat, normalize_factor, precision);

    int16_t max_class = argmax(tempMat);
    int16_t max_logit = tempMat->data[max_class];

    if (max_logit > threshold) {
        return max_class;
    }
    return -1;
}


matrix *normalize(matrix *vec, const int16_t *mean, const int16_t *std, int16_t precision) {
    /**
     * Normalizes the vector to a standard normal distribution. This operation is in-place,
     * so the original vector is mutated.
     */
    if (vec->numCols != 1) {
        return NULL_PTR;
    }

    int16_t shifted;
    uint16_t i;

    for (i = vec->numRows; i > 0; i--) {
        shifted = fp_sub(vec->data[i - 1], mean[i - 1]);
        vec->data[i - 1] = fp_div(shifted, std[i - 1], precision);
    }

    return vec;
}


matrix *hashed_matrix_vector_product(matrix *result, matrix *mat, matrix *vec, char *seed, uint8_t should_transpose, int16_t precision) {
    /**
     * Computes the matrix vector product using the hashing trick where the matrix is compressed into a vector.
     */
    // Validate dimensions.
    if ((mat->numCols != 1) || (vec->numCols != 1) | (result->numCols != 1)) {
        return NULL_PTR;
    }
    
    // Get the length of the seed
    uint16_t n = string_length(seed);

    // Create the hashing seed by pre-prending the given prefix.
    string_copy(hash_str, seed, n);
    hash_str[n + 2 * MAX_NUM_DIGITS + 1] = '\0';  // Ensure the string is null-terminated

    // Compute the matrix vector product
    uint16_t i, j, i_offset, j_offset, mat_index, num_digits;
    uint16_t first_index, second_index;
    int8_t sign;
    for (i = result->numRows; i > 0; i--) {
        i_offset = i - 1;

        result->data[i_offset] = 0;    

        for (j = vec->numRows; j > 0; j--) {
            j_offset = j - 1;

            if (should_transpose) {
                first_index = j_offset;
                second_index = i_offset;
            } else {
                first_index = i_offset;
                second_index = j_offset;
            }

            num_digits = append_int_to_str(hash_str + n, first_index);
            num_digits += append_int_to_str(hash_str + n + num_digits, second_index);
            hash_str[n + num_digits] = 's';
            hash_str[n + num_digits + 1] = '\0';

            mat_index = pearson_hash(hash_str, n + num_digits) % mat->numRows;
            sign = 2 * (pearson_hash(hash_str, n + num_digits + 1) % 2)  - 1;

            result->data[i_offset] = fp_add(sign * fp_mul(mat->data[mat_index], vec->data[j_offset], precision), result->data[i_offset]);
        }
    }

    return result;
}

