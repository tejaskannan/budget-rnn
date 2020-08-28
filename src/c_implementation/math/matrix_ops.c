#include "matrix_ops.h"


matrix *matrix_add(matrix *result, matrix *mat1, matrix *mat2) {
    /**
     * Adds two matrices together elementwise.
     * Result stored stored in the result argument (and returned for convenience).
     */
    // Validate dimensions
    if ((mat1->numRows != mat2->numRows) || (result->numRows != mat1->numRows)) {
        return NULL_PTR;
    }
    
    uint16_t rows = mat1->numRows;
    uint16_t cols = mat1->numCols > mat2->numCols ? mat2->numCols : mat1->numCols;

    uint16_t mat1Offset, mat2Offset, resultOffset;
    uint16_t rowOffset, colOffset;

    // Compute the element-wise sum. We generally add small vectors together, and the LEA
    // provides little (or negative) benefit due to added overhead.
    uint16_t i, j;
    for (i = rows; i > 0; i--) {

        rowOffset = i - 1;

        mat1Offset = rowOffset * mat1->numCols;
        mat2Offset = rowOffset * mat2->numCols;
        resultOffset = rowOffset * result->numCols;

        for (j = cols; j > 0; j--) {
            colOffset = j - 1;
            result->data[resultOffset + colOffset] = fp_add(mat1->data[mat1Offset + colOffset], mat2->data[mat2Offset + colOffset]);
        }
    }

    return result;
}


matrix *matrix_neg(matrix *result, matrix *mat, uint16_t precision) {
    /*
     * Multiplies all elements by -1.
     */
    return scalar_product(result, mat, int_to_fp(-1, precision), precision);
}


matrix *matrix_multiply(matrix *result, matrix *mat1, matrix *mat2, uint16_t precision) {
    /**
     * Performs matrix multiplication and stores value in given result array. The implementation
     * depends on whether or not we are compiling for the MSP430 device.
     */

    // Validate dimensions
    if ((mat1->numCols != mat2->numRows) || (mat1->numRows != result->numRows) || (mat2->numCols != result->numCols)) {
        return NULL_PTR;
    }

    // The result will be a [n, p] matrix
    uint16_t n = mat1->numRows;
    uint16_t m = mat1->numCols;
    uint16_t p = mat2->numCols;

    #ifdef IS_MSP
    // When using the MSP430, we use the LEA for matrix multiplications. Based on profiling,
    // the LEA can take up to 5x fewer compute cycles than a standard implementation.

    msp_status status;
    msp_matrix_mpy_q15_params mulParams;

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

    // Convert back to the original fixed-point precision. The LEA assumes 15 fractional bits.
    msp_matrix_shift_q15_params shiftParams;
    shiftParams.rows = result->numRows;
    shiftParams.cols = result->numCols;
    shiftParams.shift = 15 - precision;

    // Perform element-wise shift using the LEA
    if (shiftParams.shift > 0) {
        status = msp_matrix_shift_q15(&shiftParams, result->data, result->data);
        msp_checkStatus(status);
    }

    #else

    uint16_t i, j, k;
    uint16_t outerRow, innerRow, resultRow;
    int16_t sum, prod;

    for (i = n; i > 0; i--) {
        outerRow = (i - 1) * m;  // Offset for the i^th row

        for (j = p; j > 0; j--) {
            sum = 0;

            for (k = m; k > 0; k--) {
                innerRow = (k - 1) * p;  // Offset for the k^th row
                prod = fp_mul(mat1->data[outerRow + (k - 1)], mat2->data[innerRow + (j - 1)], precision);
                sum = fp_add(sum, prod);
            }
 
            resultRow = (i - 1) * p;
            result->data[resultRow + (j - 1)] = sum;
        }
    }
    #endif

    return result;
}


int16_t dot_product(matrix *vec1, matrix *vec2, uint16_t precision) {
    /**
     * Computes the dot product for the two vectors. If the inputs are not
     * proper vectors, then we use the first row of vec1 and first column of vec2.
     */
    uint16_t i, j;
    uint16_t vec1Idx, vec2Idx;
    int16_t result = 0;

    for (i = vec1->numCols; i > 0; i--) {
        vec1Idx = i - 1;
        vec2Idx = vec2->numCols * (i - 1);

        result = fp_add(result, fp_mul(vec1->data[vec1Idx], vec2->data[vec2Idx], precision));
    }

    return result;
}



matrix *matrix_hadamard(matrix* result, matrix *mat1, matrix *mat2, uint16_t precision) {
    /**
     * Elementwise matrix product. Result stored in matrix 1. We never use the LEA to avoid
     * added overhead (we only multiply small vectors).
     */
    // Validate dimensions
    if ((mat1->numRows != mat2->numRows) || (result->numRows != mat1->numRows)) {
        return NULL_PTR;
    }
    
    uint16_t rows = mat1->numRows;
    uint16_t cols = mat1->numCols > mat2->numCols ? mat2->numCols : mat1->numCols;

    uint16_t mat1Offset, mat2Offset, resultOffset;
    uint16_t rowOffset, colOffset;

    // Compute the element-wise sum. We generally add small vectors together, and the LEA
    // provides little (or negative) benefit due to added overhead.
    uint16_t i, j;
    for (i = rows; i > 0; i--) {

        rowOffset = i - 1;

        mat1Offset = rowOffset * mat1->numCols;
        mat2Offset = rowOffset * mat2->numCols;
        resultOffset = rowOffset * result->numCols;

        for (j = cols; j > 0; j--) {
            colOffset = j - 1;
            result->data[resultOffset + colOffset] = fp_mul(mat1->data[mat1Offset + colOffset], mat2->data[mat2Offset + colOffset], precision);
        }
    }

    return result;
}


matrix *scalar_product(matrix *result, matrix *mat, int16_t scalar, uint16_t precision) {
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


matrix *apply_elementwise(matrix *result, matrix *mat, int16_t (*fn)(int16_t, uint16_t), uint16_t precision) {
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



matrix *vstack(matrix *result, matrix *mat1, matrix *mat2) {
    /**
     * Stacks the two matrices into a larger matrix. For example, if mat1 is [n, m] and
     * mat2 is [p, m], the result must be [n + p, m] where mat is placed about mat2.
     */
    // Validate the input shapes.
    if ((mat1->numRows + mat2->numRows != result->numRows) || (mat1->numCols != mat2->numCols) ||
        (mat1->numCols != result->numCols)) {
        return NULL_PTR;
    }

    uint16_t cols = mat1->numCols;

    // Copy in the first matrix
    uint16_t i;
    for (i = mat1->numRows * cols; i > 0; i--) {
        result->data[i-1] = mat1->data[i-1];
    }

    // Copy in the second matrix
    uint16_t offset = mat1->numRows * cols;
    for (i = mat2->numRows * cols; i > 0; i--) {
        result->data[offset + i - 1] = mat2->data[i-1];
    }

    return result;
}


int16_t argmax(matrix *vec) {
    /**
     * Computes the argmax of the 1d vector. If the input has multiple columns, then
     * this is the argmax over the 1st column.
     */
    if (vec->numRows <= 0) {
        return -1;
    }

    uint16_t numCols = vec->numCols;

    int16_t max = vec->data[0];
    int16_t max_index = 0;

    uint16_t i;
    int16_t val;
    for (i = vec->numRows - 1; i > 0; i--) {
        val = vec->data[i * numCols];
        if (val > max) {
            max_index = i;
            max = val;
        }
    }

    return max_index;
}
