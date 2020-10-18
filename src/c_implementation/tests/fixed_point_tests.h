#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "../math/fixed_point_ops.h"

#ifndef FIXED_POINT_TESTS_GUARD
#define FIXED_POINT_TESTS_GUARD

void test_mul_basic();
void test_mul_neg();
void test_div_basic();
void test_mod();
void test_tanh_basic();
void test_sigmoid_basic();
void test_relu_basic();
void test_leaky_relu();
void test_round_to_int();
void test_div32();
void test_sub32();

uint8_t test_equality(int32_t expected, int32_t result);

#endif
