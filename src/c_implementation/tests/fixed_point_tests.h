#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "../math/fixed_point_ops.h"

#ifndef FIXED_POINT_TESTS_GUARD
#define FIXED_POINT_TESTS_GUARD

void test_mul_basic();
void test_div_basic();
void test_exp_basic();
void test_exp_neg();
void test_tanh_basic();
void test_sigmoid_basic();
void test_relu_basic();
void test_round_to_int();

#endif
