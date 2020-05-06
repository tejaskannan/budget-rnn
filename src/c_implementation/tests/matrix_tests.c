#include "matrix_tests.h"


int main(void) {

    // Matrix allocation
    printf("---- Testing Matrix Allocation ----\n");
    test_allocate();
    printf("\tPassed allocation tests.\n");

    // Matrix addition
    printf("---- Testing Matrix Addition ----\n");
    test_add_two();
    test_add_three();
    test_add_diff();
    test_add_wrong_dims();
    printf("\tPassed addition tests.\n");

    // Matrix Multiplication
    printf("---- Testing Matrix Multiplication ----\n");
    test_mult_two();
    test_mult_three();
    test_mult_diff();
    test_mult_vec();
    test_mult_wrong_dims();
    printf("\tPassed multiplication tests.\n");

    // Matrix Hadamard Product
    printf("---- Testing Hadamard Product ----\n");
    test_hadamard_two();
    test_hadamard_three();
    test_hadamard_diff();
    test_hadamard_wrong_dims();
    printf("\tPassed multiplication tests.\n");

    // Matrix Scalar Product
    printf("---- Testing Scalar Product ----\n");
    test_scalar_mult();
    printf("\tPassed scalar multiplication tests.\n");

    // Matrix Scalar Addition
    printf("---- Testing Scalar Addition ----\n");
    test_scalar_add();
    printf("\tPassed scalar multiplication tests.\n");

    // Matrix apply element-wise sigmoid
    printf("---- Testing Apply Element-Wise ----\n");
    test_apply_sigmoid();
    printf("\tPassed apply element-wise tests.\n");

    // Matrix Replace
    printf("---- Testing Matrix Replace ----\n");
    test_replace();
    test_replace_wrong_dims();
    printf("\tPassed replacement tests.\n");

    // Matrix Transpose
    printf("---- Testing Matrix Transpose ----\n");
    test_transpose();
    test_transpose_wrong_dims();
    printf("\tPassed transpose tests.\n");

    // Vector Argmax
    printf("---- Testing Vector Argmax ----\n");
    test_argmax();
    printf("\tPassed argmax tests.\n");

    printf("--------------------\n");
    printf("Completed all tests.\n");
    return 0;
}


void test_allocate(void) {
    int n = 2;
    int m = 3;

    matrix *mat1 = matrix_allocate(n, m);
    assert(mat1->numRows == n);
    assert(mat1->numCols == m);
    assert((2 + (n * m * sizeof(int16_t)) + sizeof(matrix)) == allocBytes());

    matrix *mat2 = matrix_allocate(2 * n, m);
    assert(mat2->numRows == 2 * n);
    assert(mat2->numCols == m);
    assert(mat1->numRows == n);
    assert(mat1->numCols == m);

    matrix_free(mat2);
    assert(mat1->numRows == n);
    assert(mat1->numCols == m);
   
    matrix *mat3 = matrix_allocate(n, 2 * m);
    assert(mat2->numRows == n);
    assert(mat2->numCols == 2 * m);
    assert(mat1->numRows == n);
    assert(mat1->numCols == m);

    matrix_free(mat3);
    matrix_free(mat1);
    assert(0 == allocBytes());
}

void test_add_two(void) {
    // Test 2 x 2 cases
    int16_t mat1Data[] = { 1, 2, 3, 4 };
    matrix *mat1 = matrix_allocate(2, 2);
    load_data(mat1, mat1Data, PRECISION);
    
    int16_t mat2Data[] = { 5, 6, 7, 8 };
    matrix *mat2 = matrix_allocate(2, 2);
    load_data(mat2, mat2Data, PRECISION);

    int16_t expectedData[] = { 6, 8, 10, 12 };
    matrix *expected = matrix_allocate(2, 2);
    load_data(expected, expectedData, PRECISION);

    matrix *sum = matrix_add(mat1, mat1, mat2);
    assert(matrix_equal(expected, sum));

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(expected);
    assert(0 == allocBytes());
}


void test_add_three(void) {
    // Test 3 x 3 cases
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    matrix *mat1 = matrix_allocate(3, 3);
    load_data(mat1, mat1Data, PRECISION);
    
    int16_t mat2Data[] = { 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    matrix *mat2 = matrix_allocate(3, 3);
    load_data(mat2, mat2Data, PRECISION);

    int16_t expectedData[] = { 12, 14, 16, 18, 20, 22, 24, 26, 28 };
    matrix *expected = matrix_allocate(3, 3);
    load_data(expected, expectedData, PRECISION);

    matrix *sum = matrix_add(mat1, mat1, mat2);
    assert(matrix_equal(expected, sum));

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(expected);
    assert(0 == allocBytes());
}

void test_add_diff(void) {
    // Test 2 x 3 * 3 x 3 case
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6 };
    matrix *mat1 = matrix_allocate(2, 3);
    load_data(mat1, mat1Data, PRECISION);

    int16_t mat2Data[] = { 7, 10, 11, 8, 12, 13 };
    matrix *mat2 = matrix_allocate(2, 3);
    load_data(mat2, mat2Data, PRECISION);

    int16_t expectedData[] = { 8, 12, 14, 12, 17, 19 };
    matrix *expected = matrix_allocate(2, 3);
    load_data(expected, expectedData, PRECISION);

    matrix *sum = matrix_add(mat1, mat1, mat2);
    assert(matrix_equal(expected, sum));

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(expected);
    assert(0 == allocBytes());
}


void test_add_wrong_dims(void) {
    // Test misaligned dimensions
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    matrix *mat1 = matrix_allocate(3, 3);
    load_data(mat1, mat1Data, PRECISION);

    int16_t mat2Data[] = { 4, 5, 6, 7 };
    matrix *mat2 = matrix_allocate(2, 2);
    load_data(mat2, mat2Data, PRECISION);

    assert(isNull(matrix_add(mat1, mat1, mat2)));
    assert(isNull(matrix_add(mat2, mat2, mat1)));

    matrix_free(mat1);
    matrix_free(mat2);
    assert(0 == allocBytes());
}


void test_mult_two(void) {
    // Test 2 x 2 cases
    int16_t mat1Data[] = { 1, 2, 3, 4 };
    matrix *mat1 = matrix_allocate(2, 2);
    load_data(mat1, mat1Data, PRECISION);
    
    int16_t mat2Data[] = { 5, 6, 7, 8 };
    matrix *mat2 = matrix_allocate(2, 2);
    load_data(mat2, mat2Data, PRECISION);

    int16_t expectedData[] = { 19, 22, 43, 50 };
    matrix *expected = matrix_allocate(2, 2);
    load_data(expected, expectedData, PRECISION);

    matrix *result = matrix_allocate(2, 2);
    result = matrix_multiply(result, mat1, mat2, PRECISION);
    assert(matrix_equal(expected, result));

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result);
    matrix_free(expected);
    assert(0 == allocBytes());
}

void test_mult_three(void) {
    // Test 3 x 3 cases
    int16_t precision = 5;

    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    matrix *mat1 = matrix_allocate(3, 3);
    load_data(mat1, mat1Data, precision);

    int16_t mat2Data[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
    matrix *mat2 = matrix_allocate(3, 3);
    load_data(mat2, mat2Data, precision);

    int16_t expectedData[] = { 84, 90, 96, 201, 216, 231, 318, 342, 366 };
    matrix *expected = matrix_create_from(expectedData, 3, 3);
    load_data(expected, expectedData, precision);

    matrix *result = matrix_allocate(3, 3);
    result = matrix_multiply(result, mat1, mat2, precision);
    assert(matrix_equal(expected, result));

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result);
    matrix_free(expected);
    assert(0 == allocBytes());
}


void test_mult_diff(void) {
    // Test 2 x 3 * 3 x 3 case
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6 };
    matrix *mat1 = matrix_allocate(2, 3);
    load_data(mat1, mat1Data, PRECISION);

    int16_t mat2Data[] = { 7, 10, 11, 8, 12, 13, 9, 14, 15 };
    matrix *mat2 = matrix_allocate(3, 3);
    load_data(mat2, mat2Data, PRECISION);

    int16_t expectedData[] = { 50, 76, 82, 122, 184, 199 };
    matrix *expected = matrix_allocate(2, 3);
    load_data(expected, expectedData, PRECISION);

    matrix *result = matrix_allocate(2, 3);
    result = matrix_multiply(result, mat1, mat2, PRECISION);
    assert(matrix_equal(expected, result));

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result);
    matrix_free(expected);
    assert(0 == allocBytes());
}


void test_mult_vec(void) {
    int16_t matData[] = { 2, 3, 4, 5, 1, 2 };
    matrix *mat = matrix_allocate(2, 3);
    load_data(mat, matData, PRECISION);

    int16_t vecData[] = { 2, 3, 4 };
    matrix *vec = matrix_allocate(3, 1);
    load_data(vec, vecData, PRECISION);

    int16_t expectedData[] = { 29, 21 };
    matrix *expected = matrix_allocate(2, 1);
    load_data(expected, expectedData, PRECISION);

    matrix *result = matrix_allocate(2, 1);
    result = matrix_multiply(result, mat, vec, PRECISION);
    assert(matrix_equal(expected, result));

    matrix_free(mat);
    matrix_free(vec);
    matrix_free(expected);
    matrix_free(result);
    assert(0 == allocBytes());
}


void test_mult_wrong_dims(void) {
    // Test cases where dimensions are not aligned
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6 };
    matrix *mat1 = matrix_allocate(2, 3);
    load_data(mat1, mat1Data, PRECISION);

    int16_t mat2Data[] = { 7, 10, 11, 12 };
    matrix *mat2 = matrix_allocate(4, 1);
    load_data(mat2, mat2Data, PRECISION);

    matrix *result = matrix_allocate(3, 3);

    assert(isNull(matrix_multiply(result, mat1, mat2, PRECISION)));
    assert(isNull(matrix_multiply(result, mat2, mat1, PRECISION)));

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(result);
    assert(0 == allocBytes());
}


void test_hadamard_two(void) {
    // Test 2 x 2 cases
    int16_t mat1Data[] = { 1, 2, 3, 4 };
    matrix *mat1 = matrix_allocate(2, 2);
    load_data(mat1, mat1Data, PRECISION);

    int16_t mat2Data[] = { 5, 6, 7, 8 };
    matrix *mat2 = matrix_allocate(2, 2);
    load_data(mat2, mat2Data, PRECISION);

    int16_t expectedData[] = { 5, 12, 21, 32 };
    matrix *expected = matrix_allocate(2, 2);
    load_data(expected, expectedData, PRECISION);

    matrix *prod = matrix_hadamard(mat1, mat1, mat2, PRECISION);
    assert(matrix_equal(expected, prod));

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(expected);
    assert(0 == allocBytes());
}


void test_hadamard_three(void) {
    // Test 3 x 3 cases
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    matrix *mat1 = matrix_allocate(3, 3);
    load_data(mat1, mat1Data, PRECISION);

    int16_t mat2Data[] = { 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    matrix *mat2 = matrix_allocate(3, 3);
    load_data(mat2, mat2Data, PRECISION);

    int16_t expectedData[] = { 11, 24, 39, 56, 75, 96, 119, 144, 171 };
    matrix *expected = matrix_allocate(3, 3);
    load_data(expected, expectedData, PRECISION);

    matrix *prod = matrix_hadamard(mat1, mat1, mat2, PRECISION);
    assert(matrix_equal(expected, prod));

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(expected);
    assert(0 == allocBytes());
}


void test_hadamard_diff(void) {
    // Test 2 x 3 * 3 x 3 case
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6 };
    matrix *mat1 = matrix_allocate(2, 3);
    load_data(mat1, mat1Data, PRECISION);

    int16_t mat2Data[] = { 7, 10, 11, 8, 12, 13 };
    matrix *mat2 = matrix_allocate(2, 3);
    load_data(mat2, mat2Data, PRECISION);

    int16_t expectedData[] = { 7, 20, 33, 32, 60, 78 };
    matrix *expected = matrix_allocate(2, 3);
    load_data(expected, expectedData, PRECISION);

    matrix *prod = matrix_hadamard(mat1, mat1, mat2, PRECISION);
    assert(matrix_equal(expected, prod));

    matrix_free(mat1);
    matrix_free(mat2);
    matrix_free(expected);
    assert(0 == allocBytes());
}


void test_hadamard_wrong_dims(void) {
    // Test misaligned dimensions
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    matrix *mat1 = matrix_allocate(3, 3);
    load_data(mat1, mat1Data, PRECISION);

    int16_t mat2Data[] = { 4, 5, 6, 7 };
    matrix *mat2 = matrix_allocate(2, 2);
    load_data(mat2, mat2Data, PRECISION);

    assert(isNull(matrix_hadamard(mat1, mat1, mat2, PRECISION)));
    assert(isNull(matrix_hadamard(mat2, mat2, mat1, PRECISION)));

    matrix_free(mat1);
    matrix_free(mat2);
    assert(0 == allocBytes());
}


void test_scalar_mult(void) {
    int16_t matData[] = { 1, 2, 3, 4, 5, 6 };
    matrix *mat = matrix_allocate(2, 3);
    load_data(mat, matData, PRECISION);

    int16_t scalar = int_to_fp(-3, PRECISION);

    int16_t expectedData[] = { -3, -6, -9, -12, -15, -18 };
    matrix *expected = matrix_allocate(2, 3);
    load_data(expected, expectedData, PRECISION);

    matrix *result = scalar_product(mat, mat, scalar, PRECISION);
    assert(matrix_equal(expected, result));

    matrix_free(mat);
    matrix_free(expected);
    assert(0 == allocBytes());
}


void test_scalar_add(void) {
    int16_t matData[] = { 1, 2, 3, 4, 5, 6 };
    matrix *mat = matrix_allocate(2, 3);
    load_data(mat, matData, PRECISION);

    int16_t scalar = int_to_fp(-4, PRECISION);

    int16_t expectedData[] = { -3, -2, -1, 0, 1, 2 };
    matrix *expected = matrix_allocate(2, 3);
    load_data(expected, expectedData, PRECISION);

    matrix *result = scalar_add(mat, mat, scalar);
    assert(matrix_equal(expected, result));

    matrix_free(mat);
    matrix_free(expected);
    assert(0 == allocBytes());
}


void test_apply_sigmoid(void) {
    int16_t matData[] = { 0, 0, 0, 0, 0, 0 };
    matrix *mat = matrix_allocate(2, 3);
    load_data(mat, matData, PRECISION);

    int16_t one_half = float_to_fp(0.5, PRECISION);
    matrix *expected = matrix_allocate(2, 3);
    for (int16_t i = 0; i < 6; i++) {
        expected->data[i] = one_half;
    }

    matrix *result = apply_elementwise(mat, mat, &fp_sigmoid, PRECISION);
    assert(matrix_equal(expected, result));

    matrix_free(mat);
    matrix_free(expected);
    assert(0 == allocBytes());
}


void test_replace(void) {
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6 };
    matrix *mat1 = matrix_allocate(2, 3);
    load_data(mat1, mat1Data, PRECISION);

    matrix *mat2 = matrix_allocate(2, 3);
    
    matrix_replace(mat2, mat1);
    assert(matrix_equal(mat1, mat2));

    matrix_free(mat1);
    matrix_free(mat2);
    assert(0 == allocBytes());
}


void test_replace_wrong_dims(void) {
    // Test misaligned dimensions
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    matrix *mat1 = matrix_allocate(3, 3);
    load_data(mat1, mat1Data, PRECISION);

    matrix *mat2 = matrix_allocate(2, 3);
    assert(isNull(matrix_replace(mat2, mat1)));

    matrix_free(mat1);
    matrix_free(mat2);
    assert(0 == allocBytes());
}


void test_transpose(void) {
    // Test matrix transpose
    int16_t matData[] = { 1, 2, 3, 4, 5, 6 };
    matrix *mat = matrix_allocate(2, 3);
    load_data(mat, matData, PRECISION);

    int16_t expectedData[] = { 1, 4, 2, 5, 3, 6 };
    matrix *expected = matrix_allocate(3, 2);
    load_data(expected, expectedData, PRECISION);

    matrix *result = matrix_allocate(3, 2);
    result = transpose(result, mat);
    assert(matrix_equal(expected, result));

    matrix_free(expected);
    matrix_free(result);
    matrix_free(mat);
    assert(0 == allocBytes());
}


void test_argmax(void) {
    int16_t vecData[] = { 2, 3, 1, 4, 5, 2 };
    matrix *vec = matrix_allocate(6, 1);
    load_data(vec, vecData, PRECISION);

    assert(4 == argmax(vec));
    
    matrix_free(vec);
    assert(0 == allocBytes());
}


void test_transpose_wrong_dims(void) {
    // Test matrix transpose with incorrect dimensions
    int16_t matData[] = { 1, 2, 3, 4, 5, 6 };
    matrix *mat = matrix_create_from(matData, 2, 3);
    load_data(mat, matData, PRECISION);

    matrix *result = matrix_allocate(2, 2);
    assert(isNull(transpose(result, mat))); 
    
    matrix_free(result);
    matrix_free(mat);
    assert(0 == allocBytes());
}


int matrix_equal(matrix *mat1, matrix *mat2) {
    if (mat1 == NULL_PTR && mat2 == NULL_PTR) {
        return 1;
    }

    if (mat1 == NULL_PTR && mat2 != NULL_PTR) {
        return 0;
    }

    if (mat1 != NULL_PTR && mat2 == NULL_PTR) {
        return 0;
    }
    
    if ((mat1->numRows != mat2->numRows) || (mat1->numCols != mat2->numCols)) {
        return 0;
    }

    for (int i = 0; i < mat1->numRows * mat2->numCols; i++) {
        if (mat1->data[i] != mat2->data[i]) {
            return 0;
        }
    }

    return 1;
}


void load_data(matrix *mat, int16_t *data, int16_t precision) {
    for (int16_t i = 0; i < mat->numRows * mat->numCols; i++) {
        mat->data[i] = int_to_fp(data[i], precision);
    }
}
