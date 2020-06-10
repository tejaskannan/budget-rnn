#include <stdio.h>
#include <assert.h>
#include <stdint.h>

#include "../memory.h"
#include "../math/matrix.h"
#include "../math/matrix_ops.h"


#ifndef MATRIX_TEST_GUARD
    #define MATRIX_TEST_GUARD

    #define PRECISION 7

    // Utility Function
    int matrix_equal(matrix *mat1, matrix *mat2);
    void load_data(matrix *mat, int16_t *data, int16_t precision);

    void test_allocate(void);
    void test_add_two(void);
    void test_add_three(void);
    void test_add_diff(void);
    void test_add_wrong_dims(void);
    void test_mult_two(void);
    void test_mult_three(void);
    void test_mult_diff(void);
    void test_mult_vec(void);
    void test_mult_wrong_dims(void);
    void test_hadamard_two(void);
    void test_hadamard_three(void);
    void test_hadamard_diff(void);
    void test_hadamard_wrong_dims(void);
    void test_scalar_mult(void);
    void test_scalar_add(void);
    void test_apply_sigmoid(void);
    void test_transpose(void);
    void test_transpose_wrong_dims(void);
    void test_replace(void);
    void test_replace_wrong_dims(void);
    void test_argmax(void);
    void test_hashed_prod_one(void);
    void test_hashed_prod_two(void);
    void test_hashed_prod_three(void);
    void test_hashed_prod_four(void);

#endif
