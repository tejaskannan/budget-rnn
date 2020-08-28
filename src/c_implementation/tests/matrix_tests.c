#include "matrix_tests.h"


int main(void) {

    // Matrix addition
    printf("---- Testing Matrix Addition ----\n");
    test_add_two();
    test_add_three();
    test_add_diff();
    test_add_min_dims();
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

    // Vector Dot Product
    printf("---- Testing Vector Dot Product ----\n");
    test_dot_product();
    test_dot_product_two();
    printf("\tPassed dot product tests.\n");

    // Matrix Hadamard Product
    printf("---- Testing Hadamard Product ----\n");
    test_hadamard_two();
    test_hadamard_three();
    test_hadamard_diff();
    test_hadamard_min_dims();
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

    // Matrix v-stack
    printf("---- Testing vstack ----\n");
    test_vstack();
    test_vstack_wrong_dims();
    printf("\tPassed vstack tests.\n");

    // Matrix apply element-wise sigmoid
    printf("---- Testing Apply Element-Wise ----\n");
    test_apply_sigmoid();
    printf("\tPassed apply element-wise tests.\n");

    // Matrix Replace
    printf("---- Testing Matrix Replace ----\n");
    test_replace();
    test_replace_wrong_dims();
    printf("\tPassed replacement tests.\n");

    // Vector Argmax
    printf("---- Testing Vector Argmax ----\n");
    test_argmax();
    test_argmax_two();
    printf("\tPassed argmax tests.\n");

    // Matrix Minimum
    printf("---- Testing Matrix Min ----\n");
    test_matrix_min();
    printf("\tPassed matrix min test.\n");

    // Matrix Sum
    printf("---- Testing Matrix Sum ----\n");
    test_matrix_sum();
    printf("\tPassed matrix sum test.\n");

    printf("--------------------\n");
    printf("Completed all tests.\n");
    return 0;
}


void test_add_two(void) {
    // Test 2 x 2 cases
    int16_t mat1Data[] = { 1, 2, 3, 4 };
    matrix mat1 = { to_fixed_point(mat1Data, 4, PRECISION), 2, 2};
    
    int16_t mat2Data[] = { 5, 6, 7, 8 };
    matrix mat2 = { to_fixed_point(mat2Data, 4, PRECISION), 2, 2 };

    int16_t expectedData[] = { 6, 8, 10, 12 };
    matrix expected = { to_fixed_point(expectedData, 4, PRECISION), 2, 2 };

    matrix *sum = matrix_add(&mat1, &mat1, &mat2);
    assert(matrix_equal(&expected, sum));
}


void test_add_three(void) {
    // Test 3 x 3 cases
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    matrix mat1 = { to_fixed_point(mat1Data, 9, PRECISION), 3, 3 };
    
    int16_t mat2Data[] = { 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    matrix mat2 = { to_fixed_point(mat2Data, 9, PRECISION), 3, 3 };

    int16_t expectedData[] = { 12, 14, 16, 18, 20, 22, 24, 26, 28 };
    matrix expected = { to_fixed_point(expectedData, 9, PRECISION), 3, 3 };

    matrix *sum = matrix_add(&mat1, &mat1, &mat2);
    assert(matrix_equal(&expected, sum));
}

void test_add_diff(void) {
    // Test 2 x 3 * 3 x 3 case
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6 };
    matrix mat1 = { to_fixed_point(mat1Data, 6, PRECISION), 2, 3 };

    int16_t mat2Data[] = { 7, 10, 11, 8, 12, 13 };
    matrix mat2 = { to_fixed_point(mat2Data, 6, PRECISION), 2, 3 };

    int16_t expectedData[] = { 8, 12, 14, 12, 17, 19 };
    matrix expected = { to_fixed_point(expectedData, 6, PRECISION), 2, 3 };

    matrix *sum = matrix_add(&mat1, &mat1, &mat2);
    assert(matrix_equal(&expected, sum));
}


void test_add_min_dims(void) {
    // Test 4 x 1 + 4 x 2 case (only adds first column)
    int16_t mat1Data[] = { 1, 2, 3, 4 };
    matrix mat1 = { to_fixed_point(mat1Data, 4, PRECISION), 4, 1 };

    int16_t mat2Data[] = { 7, 10, 11, 8, 12, 13, 15, 3 };
    matrix mat2 = { to_fixed_point(mat2Data, 8, PRECISION), 4, 2 };

    int16_t expectedData[] = { 8, 13, 15, 19 };
    matrix expected = { to_fixed_point(expectedData, 4, PRECISION), 4, 1 };

    matrix *sum = matrix_add(&mat1, &mat1, &mat2);
    assert(matrix_equal(&expected, sum));
}


void test_add_wrong_dims(void) {
    // Test misaligned dimensions
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    matrix mat1 = { to_fixed_point(mat1Data, 9, PRECISION), 3, 3 };

    int16_t mat2Data[] = { 4, 5, 6, 7 };
    matrix mat2 = { to_fixed_point(mat2Data, 4, PRECISION), 2, 2 };

    assert(matrix_add(&mat1, &mat1, &mat2) == NULL_PTR);
    assert(matrix_add(&mat2, &mat2, &mat1) == NULL_PTR);
}


void test_mult_two(void) {
    // Test 2 x 2 cases
    int16_t mat1Data[] = { 1, 2, 3, 4 };
    matrix mat1 = { to_fixed_point(mat1Data, 4, PRECISION), 2, 2 };

    int16_t mat2Data[] = { 5, 6, 7, 8 };
    matrix mat2 = { to_fixed_point(mat2Data, 4, PRECISION), 2, 2 };

    int16_t expectedData[] = { 19, 22, 43, 50 };
    matrix expected = { to_fixed_point(expectedData, 4, PRECISION), 2, 2 };

    int16_t resultData[4] = { 0 };
    matrix result = { resultData, 2, 2 };

    matrix_multiply(&result, &mat1, &mat2, PRECISION);
    assert(matrix_equal(&expected, &result));
}

void test_mult_three(void) {
    // Test 3 x 3 cases
    int16_t precision = 5;

    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    matrix mat1 = { to_fixed_point(mat1Data, 9, precision), 3, 3 };

    int16_t mat2Data[] = { 10, 11, 12, 13, 14, 15, 16, 17, 18 };
    matrix mat2 = { to_fixed_point(mat2Data, 9, precision), 3, 3 };

    int16_t expectedData[] = { 84, 90, 96, 201, 216, 231, 318, 342, 366 };
    matrix expected = { to_fixed_point(expectedData, 9, precision), 3, 3 };

    int16_t resultData[9] = { 0 };
    matrix result = { resultData, 3, 3 };
    
    matrix_multiply(&result, &mat1, &mat2, precision);
    assert(matrix_equal(&expected, &result));
}


void test_mult_diff(void) {
    // Test 2 x 3 * 3 x 3 case
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6 };
    matrix mat1 = { to_fixed_point(mat1Data, 6, PRECISION), 2, 3 };

    int16_t mat2Data[] = { 7, 10, 11, 8, 12, 13, 9, 14, 15 };
    matrix mat2 = { to_fixed_point(mat2Data, 9, PRECISION), 3, 3 };

    int16_t expectedData[] = { 50, 76, 82, 122, 184, 199 };
    matrix expected = { to_fixed_point(expectedData, 6, PRECISION), 2, 3 };

    int16_t resultData[6] = { 0 };
    matrix result = { resultData, 2, 3 };

    matrix_multiply(&result, &mat1, &mat2, PRECISION);
    assert(matrix_equal(&expected, &result));
}


void test_mult_vec(void) {
    int16_t matData[] = { 2, 3, 4, 5, 1, 2 };
    matrix mat = { to_fixed_point(matData, 6, PRECISION), 2, 3 };

    int16_t vecData[] = { 2, 3, 4 };
    matrix vec = { to_fixed_point(vecData, 3, PRECISION), 3, 1 };

    int16_t expectedData[] = { 29, 21 };
    matrix expected = { to_fixed_point(expectedData, 2, PRECISION), 2, 1 };

    int16_t resultData[2] = { 0 };
    matrix result = { resultData, 2, 1 };

    matrix_multiply(&result, &mat, &vec, PRECISION);
    assert(matrix_equal(&expected, &result));
}


void test_mult_wrong_dims(void) {
    // Test cases where dimensions are not aligned
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6 };
    matrix mat1 = { mat1Data, 2, 3 };

    int16_t mat2Data[] = { 7, 10, 11, 12 };
    matrix mat2 = { mat2Data, 4, 1 };

    int16_t resultData[9] = { 0 };
    matrix result = { resultData, 3, 3 };

    assert(matrix_multiply(&result, &mat1, &mat2, PRECISION) == NULL_PTR);
    assert(matrix_multiply(&result, &mat2, &mat1, PRECISION) == NULL_PTR);
}


void test_hadamard_two(void) {
    // Test 2 x 2 cases
    int16_t mat1Data[] = { 1, 2, 3, 4 };
    matrix mat1 = { to_fixed_point(mat1Data, 4, PRECISION), 2, 2 };

    int16_t mat2Data[] = { 5, 6, 7, 8 };
    matrix mat2 = { to_fixed_point(mat2Data, 4, PRECISION), 2, 2 };

    int16_t expectedData[] = { 5, 12, 21, 32 };
    matrix expected = { to_fixed_point(expectedData, 4, PRECISION), 2, 2 };

    matrix *prod = matrix_hadamard(&mat1, &mat1, &mat2, PRECISION);
    assert(matrix_equal(&expected, prod));
}


void test_hadamard_three(void) {
    // Test 3 x 3 cases
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    matrix mat1 = { to_fixed_point(mat1Data, 9, PRECISION), 3, 3 };

    int16_t mat2Data[] = { 11, 12, 13, 14, 15, 16, 17, 18, 19 };
    matrix mat2 = { to_fixed_point(mat2Data, 9, PRECISION), 3, 3 };

    int16_t expectedData[] = { 11, 24, 39, 56, 75, 96, 119, 144, 171 };
    matrix expected = { to_fixed_point(expectedData, 9, PRECISION), 3, 3 };

    matrix *prod = matrix_hadamard(&mat1, &mat1, &mat2, PRECISION);
    assert(matrix_equal(&expected, prod));
}


void test_hadamard_diff(void) {
    // Test 2 x 3 * 3 x 3 case
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6 };
    matrix mat1 = { to_fixed_point(mat1Data, 6, PRECISION), 2, 3 };

    int16_t mat2Data[] = { 7, 10, 11, 8, 12, 13 };
    matrix mat2 = { to_fixed_point(mat2Data, 6, PRECISION), 2, 3 };

    int16_t expectedData[] = { 7, 20, 33, 32, 60, 78 };
    matrix expected = { to_fixed_point(expectedData, 6, PRECISION), 2, 3 };

    matrix *prod = matrix_hadamard(&mat1, &mat1, &mat2, PRECISION);
    assert(matrix_equal(&expected, prod));
}


void test_hadamard_min_dims(void) {
    // Test 4 x 1 * 4 x 2 case (only multiply the first column)
    int16_t mat1Data[] = { 1, 2, 3, 4 };
    matrix mat1 = { to_fixed_point(mat1Data, 4, PRECISION), 4, 1 };

    int16_t mat2Data[] = { 5, 2, 1, 7, 4, 9, 2, 10 };
    matrix mat2 = { to_fixed_point(mat2Data, 8, PRECISION), 4, 2 };

    int16_t expectedData[] = { 5, 2, 2, 7, 12, 9, 8, 10  };
    matrix expected = { to_fixed_point(expectedData, 8, PRECISION), 4, 2 };

    matrix *prod = matrix_hadamard(&mat2, &mat1, &mat2, PRECISION);
    assert(matrix_equal(&expected, prod));
}


void test_hadamard_wrong_dims(void) {
    // Test misaligned dimensions
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    matrix mat1 = { mat1Data, 3, 3 };

    int16_t mat2Data[] = { 4, 5, 6, 7 };
    matrix mat2 = { mat2Data, 2, 2 };

    assert(matrix_hadamard(&mat1, &mat1, &mat2, PRECISION) == NULL_PTR);
    assert(matrix_hadamard(&mat2, &mat2, &mat1, PRECISION) == NULL_PTR);
}


void test_scalar_mult(void) {
    int16_t matData[] = { 1, 2, 3, 4, 5, 6 };
    matrix mat = { to_fixed_point(matData, 6, PRECISION), 2, 3 };

    int16_t scalar = int_to_fp(-3, PRECISION);

    int16_t expectedData[] = { -3, -6, -9, -12, -15, -18 };
    matrix expected = { to_fixed_point(expectedData, 6, PRECISION), 2, 3 };

    matrix *result = scalar_product(&mat, &mat, scalar, PRECISION);
    assert(matrix_equal(&expected, result));
}


void test_scalar_add(void) {
    int16_t matData[] = { 1, 2, 3, 4, 5, 6 };
    matrix mat = { to_fixed_point(matData, 6, PRECISION), 2, 3 };

    int16_t scalar = int_to_fp(-4, PRECISION);

    int16_t expectedData[] = { -3, -2, -1, 0, 1, 2 };
    matrix expected = { to_fixed_point(expectedData, 6, PRECISION), 2, 3 };

    matrix *result = scalar_add(&mat, &mat, scalar);
    assert(matrix_equal(&expected, result));
}

void test_vstack(void) {
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6 };
    matrix mat1 = { mat1Data, 3, 2 };

    int16_t mat2Data[] = { -1, -2, -3, -4 };
    matrix mat2 = { mat2Data, 2, 2 };

    int16_t resultData[10] = { 0 };
    matrix result = { resultData, 5, 2 };

    int16_t expectedData[10] = {  1, 2, 3, 4, 5, 6, -1, -2, -3, -4 };
    matrix expected = { expectedData, 5, 2 };

    vstack(&result, &mat1, &mat2);
    assert(matrix_equal(&expected, &result));
}

void test_vstack_wrong_dims(void) {
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    matrix mat1 = { mat1Data, 3, 3 };

    int16_t mat2Data[] = { -1, -2, -3, -4 };
    matrix mat2 = { mat2Data, 2, 2 };

    int16_t resultData[10] = { 0 };
    matrix result = { resultData, 5, 2 };

    assert(vstack(&result, &mat1, &mat2) == NULL_PTR);
    assert(vstack(&result, &mat2, &mat2) == NULL_PTR);
}

void test_apply_sigmoid(void) {
    int16_t matData[6] = { 0 };
    matrix mat = { matData, 2, 3 };

    int16_t one_half = float_to_fp(0.5, PRECISION);
    int16_t expectedData[6];
    for (int16_t i = 0; i < 6; i++) {
        expectedData[i] = one_half;
    }
    matrix expected = { expectedData, 2, 3 };

    matrix *result = apply_elementwise(&mat, &mat, &fp_sigmoid, PRECISION);
    assert(matrix_equal(&expected, result));
}


void test_replace(void) {
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6 };
    matrix mat1 = { to_fixed_point(mat1Data, 6, PRECISION), 2, 3 };

    int16_t mat2Data[6] = { 0 };
    matrix mat2 = { mat2Data, 2, 3 };
    
    matrix_replace(&mat2, &mat1);
    assert(matrix_equal(&mat1, &mat2));
}


void test_replace_wrong_dims(void) {
    // Test misaligned dimensions
    int16_t mat1Data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    matrix mat1 = { to_fixed_point(mat1Data, 9, PRECISION), 3, 3 };

    int16_t mat2Data[6] = { 0 };
    matrix mat2 = { mat2Data, 2, 3 };
    assert(matrix_replace(&mat2, &mat1) == NULL_PTR);
}


void test_argmax(void) {
    int16_t vecData[] = { 2, 3, 1, 4, 5, 2 };
    matrix vec = { to_fixed_point(vecData, 6, PRECISION), 6, 1 };
    assert(4 == argmax(&vec));
}


void test_argmax_two(void) {
    int16_t vecData[] = { 2, 3, 1, 4, 5, 6 };
    matrix vec = { to_fixed_point(vecData, 6, PRECISION), 3, 2 };
    assert(2 == argmax(&vec));
}


void test_dot_product(void) {
    int16_t vec1Data[] = { 2, 3, 1 };
    matrix vec1 = { to_fixed_point(vec1Data, 3, PRECISION), 1, 3 };

    int16_t vec2Data[] = { 4, 5, 2 };
    matrix vec2 = { to_fixed_point(vec2Data, 3, PRECISION), 3, 1 };

    int16_t result = dot_product(&vec1, &vec2, PRECISION);
    assert(int_to_fp(25, PRECISION) == result);
}


void test_dot_product_two(void) {
    int16_t vec1Data[] = { 2, 3, 1, 1, 1, 1 };
    matrix vec1 = { to_fixed_point(vec1Data, 3, PRECISION), 1, 3 };

    int16_t vec2Data[] = { 4, 1, 5, 1, 2, 1 };
    matrix vec2 = { to_fixed_point(vec2Data, 6, PRECISION), 3, 2 };

    int16_t result = dot_product(&vec1, &vec2, PRECISION);
    assert(int_to_fp(25, PRECISION) == result);
}


void test_matrix_sum(void) {
    int16_t matData[] = { 2, 3, 1, 4, 5, 2 };
    matrix mat = { to_fixed_point(matData, 6, PRECISION), 3, 2 };

    int16_t sum = matrix_sum(&mat);
    assert(int_to_fp(17, PRECISION) == sum);
}


void test_matrix_min(void) {
    int16_t matData[] = { 2, 3, 1, 4, 5, 2 };
    matrix mat = { to_fixed_point(matData, 6, PRECISION), 3, 2 };

    int16_t min = matrix_min(&mat);
    assert(int_to_fp(1, PRECISION) == min);
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


int16_t *to_fixed_point(int16_t *data, uint16_t n, uint16_t precision) {
    for (int16_t i = 0; i < n; i++) {
        data[i] = int_to_fp(data[i], precision);
    }

    return data;
}
