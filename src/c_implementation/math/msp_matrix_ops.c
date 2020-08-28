#include "matrix_ops.h"

static char hash_str[100];


matrix *matrix_add(matrix *result, matrix *mat1, matrix *mat2) {
    /**
     * Adds two matrices together elementwise.
     * Result stored in the matrix result. This matrix is also returned
     * for convenience.
     */
    
    // Validate dimensions
    if ((mat1->numRows != mat2->numRows) || (mat1->numCols != mat2->numCols) ||
        (result->numRows != mat1->numRows) || (result->numCols != mat2->numCols)) {
        return (matrix *) NULL_PTR;
    }
    
    // Initialize LEA metadata
    msp_status status;
    msp_matrix_add_q15_params addParams;
    addParams.rows = mat1->numRows;
    addParams.cols = mat1->numCols;

    // Execute the addition using the LEA
    status = msp_matrix_add_q15(&addParams, mat1->data, mat2->data, result->data);
    msp_checkStatus(status);

    return result;
}


matrix *matrix_multiply(matrix *result, matrix *mat1, matrix *mat2, int16_t precision) {
    /**
     * Performs matrix multiplication and stores value in given result array
     */

    // Validate dimensions
    if ((mat1->numCols != mat2->numRows) || (mat1->numRows != result->numRows) || (mat2->numCols != result->numCols)) {
        return (matrix *) NULL_PTR;
    }

    // Initialze LEA metadata
    msp_status status;
    msp_matrix_mpy_q15_params mulParams;
    mulParams.srcARows = mat1->numRows;
    mulParams.srcACols = mat1->numCols;
    mulParams.srcBRows = mat2->numRows;
    mulParams.srcBCols = mat2->numCols;

    // Perform matrix multiplication using the LEA
    status = msp_matrix_mpy_q15(&mulParams, mat1->data, mat2->data, result->data);
    msp_checkStatus(status);

    // Convert back to the original fixed-point precision
    msp_matrix_shift_q15_params shiftParams;
    shiftParams.rows = result->numRows;
    shiftParams.cols = result->numCols;
    shiftParams.shift = (int8_t) (15 - precision);

    // Perform element-wise shift using the LEA
    status = msp_matrix_shift_q15(&shiftParams, result->data, result->data);
    msp_checkStatus(status);

    return result;
}


matrix *matrix_hadamard(matrix* result, matrix *mat1, matrix *mat2, int16_t precision) {
    /**
     * Elementwise matrix product. Result stored in the result parameter (also returned for convenience).
     */
    
    // Validate dimensions
    if ((mat1->numRows != mat2->numRows) || (mat1->numCols != mat2->numCols) ||
        (result->numRows != mat1->numRows) || (result->numCols != mat1->numCols)) {
        return (matrix *) NULL_PTR;
    }

    // Initialize the LEA parameters
    msp_status status;
    msp_mpy_q15_params mpyParams;
    mpyParams.length = mat1->numRows * mat1->numCols;  // Treat the matrices as vectors

    // Perform the element-wise multiplication
    status = msp_mpy_q15(&mpyParams, mat1->data, mat2->data, result->data);

    // Convert back to the original fixed-point precision
    msp_matrix_shift_q15_params shiftParams;
    shiftParams.rows = result->numRows;
    shiftParams.cols = result->numCols;
    shiftParams.shift = (int8_t) (15 - precision);

    // Perform element-wise shift using the LEA
    status = msp_matrix_shift_q15(&shiftParams, result->data, result->data);
    msp_checkStatus(status);

    return result;
}


matrix *scalar_product(matrix *result, matrix *mat, int16_t scalar, int16_t precision) {
    /**
     * Multiplies every element in the matrix by the given scalar.
     * Result stored directly into the given array.
     */
    // Validate dimensions
    if ((result->numRows != mat->numRows) || (result->numCols != mat->numCols)) {
        return (matrix *) NULL_PTR;
    }

    // Initialize LEA parameters
    msp_status status;
    msp_matrix_scale_q15_params scaleParams;
    scaleParams.rows = mat->numRows;
    scaleParams.cols = mat->numCols;
    scaleParams.shift = 15 - precision;  // 15 is the fixed point precision of Q15 values
    scaleParams.scale = scalar;

    // Execute the addition operation using the LEA. As a note, scaling doesn't actually
    // use the LEA internally. We use this for correctness / consistency purposes.
    status = msp_matrix_scale_q15(&scaleParams, mat->data, result->data);
    msp_checkStatus(status);

    return result;
}


matrix *scalar_add(matrix *result, matrix *mat, int16_t scalar) {
    /**
     * Adds the given scalar to every element of the matrix.
     * Stores result directly in the matrix.
     */
    // Validate dimensions
    if ((result->numRows != mat->numRows) || (result->numCols != mat->numCols)) {
        return (matrix *) NULL_PTR;
    }
    
    // Initialize LEA parameters
    msp_status status;
    msp_matrix_offset_q15_params offsetParams;
    offsetParams.rows = mat->numRows;
    offsetParams.cols = mat->numCols;
    offsetParams.offset = scalar;

    // Execute the addition operation using the LEA
    status = msp_matrix_offset_q15(&offsetParams, mat->data, result->data);
    msp_checkStatus(status);

    return result;
}


matrix *apply_elementwise(matrix *result, matrix *mat, int16_t (*fn)(int16_t, int16_t), int16_t precision) {
    /**
     * Applies the given function to every element of the
     * input matrix. Result stored directly in the matrix.
     */
    // Validate dimensions
    if ((result->numRows != mat->numRows) || (result->numCols != mat->numCols)) {
        return (matrix *) NULL_PTR;
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
        return (matrix *) NULL_PTR;
    }

    // Initialize LEA parameters
    msp_status status;
    msp_matrix_trans_q15_params trParams;
    trParams.rows = mat->numRows;
    trParams.cols = mat->numCols;

    // Execute the addition operation using the LEA
    status = msp_matrix_trans_q15(&trParams, mat->data, result->data);
    msp_checkStatus(status);

    return result;
}


matrix *matrix_replace(matrix *dst, matrix *src) {
    /**
     * Replaces the contents of the destination matrix with those from the src.
     */
    if ((dst->numRows != src->numRows) || (dst->numCols != src->numCols)) {
        return (matrix *) NULL_PTR;
    }

    uint16_t i;
    for (i = dst->numRows * dst->numCols; i > 0; i--) {
        dst->data[i - 1] = src->data[i - 1];
    }

    return dst;
}


matrix *matrix_subtract(matrix *result, matrix *mat1, matrix *mat2) {
    /**
     * Performs element-wise subtraction and places the result in the destination matrix.
     */

    // Initialize LEA parameters
    msp_status status;
    msp_matrix_sub_q15_params subParams;
    subParams.rows = mat1->numRows;
    subParams.cols = mat1->numCols;

    // Execute the addition operation using the LEA
    status = msp_matrix_sub_q15(&subParams, mat1->data, mat2->data, result->data);
    msp_checkStatus(status);

    return result;
}


matrix *matrix_neg(matrix *result, matrix *mat) {
    /**
     * Negates all elements in the matrix (mat) and places the result in the destination matrix (result).
     */

    // Initialize LEA parameters
    msp_status status;
    msp_matrix_neg_q15_params negParams;
    negParams.rows = mat->numRows;
    negParams.cols = mat->numCols;

    // Execute the addition operation using the LEA
    status = msp_matrix_neg_q15(&negParams, mat->data, result->data);
    msp_checkStatus(status);

    return result;
}


matrix *matrix_set(matrix *mat, int16_t value) {
    /**
     * Sets all values in the matrix to the given value (already in fixed point form).
     */

    // Initialize LEA parameters
    msp_status status;
    msp_fill_q15_params fillParams;
    fillParams.length = mat->numRows * mat->numCols;
    fillParams.value = value;

    // Execute the set operation using the LEA
    status = msp_fill_q15(&fillParams, mat->data);
    msp_checkStatus(status);

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

    // Initialize LEA parameters
    msp_status status;
    msp_min_q15_params minParams;
    minParams.length = mat->numRows * mat->numCols;

    // Execute the min operation using the LEA
    int16_t minVal;
    uint16_t minIndex;
    status = msp_min_q15(&minParams, mat->data, &minVal, &minIndex);
    msp_checkStatus(status);

    return minVal;
}



matrix *stack(matrix *result, matrix *vec1, matrix *vec2) {
    /**
     * Stacks the two vectors into a larger vector.
     */
    // Validate the input shapes.
    if ((vec1->numRows + vec2->numRows != result->numRows) || (vec1->numCols != 1) ||
        (vec2->numCols != 1) || (result->numCols != 1)) {
        return (matrix *) NULL_PTR;
    }

    // Initialize LEA parameters
    msp_status status;
    msp_copy_q15_params copyParams;

    // Use LEA to copy in the first vector
    copyParams.length = vec1->numRows;
    status = msp_copy_q15(&copyParams, vec1->data, result->data);
    msp_checkStatus(status);

    // Use LEA to  copy in the second vector (place after vec1's data)
    copyParams.length = vec2->numRows;
    status = msp_copy_q15(&copyParams, vec2->data, result->data + vec1->numRows);
    msp_checkStatus(status);

    return result;
}


int16_t argmax(matrix *vec) {
    /**
     * Computes the argmax of the 1d vector.
     */
    if (vec->numCols != 1 || vec->numRows <= 0) {
        return -1;
    }

    // Initialize LEA parameters
    msp_status status;
    msp_max_q15_params maxParams;
    maxParams.length = vec->numRows;

    // Execute the max operation using the LEA
    int16_t maxVal;
    uint16_t maxIndex;
    status = msp_max_q15(&maxParams, vec->data, &maxVal, &maxIndex);
    msp_checkStatus(status);

    return (int16_t) maxIndex;
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
    // dtype data[logits->numRows * logits->numCols];
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
    if (max_class < 0) {
        return -1;
    }

    int16_t max_logit = tempMat->data[(uint16_t) max_class];

    if (max_logit > threshold) {
        return max_class;
    }
    return -1;
}


matrix *normalize(matrix *vec, const int16_t *mean, const int16_t *inv_std, int16_t precision) {
    /**
     * Normalizes the vector to a standard normal distribution. This operation is in-place,
     * so the original vector is mutated.
     * As a note, the inv_std should be the reciprocal of the standard deviation. By using the reciprocal
     * instead of the original, we can perform normalization by multiplication instead of division. This is more efficient.
     */
    // Validate the dimensions
    if (vec->numCols != 1) {
        return (matrix *) NULL_PTR;
    }

    // Initialize LEA Parameters for mean shifting
    msp_status status;
    msp_sub_q15_params subParams;
    subParams.length = vec->numRows;

    status = msp_sub_q15(&subParams, vec->data, mean, vec->data);
    msp_checkStatus(status);

    // Perform the std dev scaling via an element-wise product
    matrix stdVec;
    stdVec.data = (dtype *) inv_std;
    stdVec.numRows = vec->numRows;
    stdVec.numCols = 1;
    vec = matrix_hadamard(vec, vec, &stdVec, precision);

    return vec;
}


matrix *hashed_matrix_vector_product(matrix *result, matrix *mat, matrix *vec, char *seed, uint8_t should_transpose, int16_t precision) {
    /**
     * Computes the matrix vector product using the hashing trick where the matrix is compressed into a vector.
     */
    // Validate dimensions.
    if ((mat->numCols != 1) || (vec->numCols != 1) | (result->numCols != 1)) {
        return (matrix *) NULL_PTR;
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

