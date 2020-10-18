#include "fixed_point_tests.h"


int main(void) {
    // Run all tests
    printf("Testing Multiply...\n");
    test_mul_basic();
    test_mul_neg();

    printf("Testing Division...\n");
    test_div_basic();

    printf("Testing Mod...\n");
    test_mod();

    printf("Testing tanh...\n");
    test_tanh_basic();

    printf("Testing sigmoid...\n");
    test_sigmoid_basic();

    printf("Testing Relu...\n");
    test_relu_basic();

    printf("Testing Round...\n");
    test_round_to_int();

    printf("Testing Leaky Relu...\n");
    test_leaky_relu();

    printf("Testing Sub32...\n");
    test_sub32();

    printf("Testing Div32...\n");
    test_div32();
}

void test_mul_basic(void) {
    int fixed_point_bits = 3;
    int16_t one = 1 << fixed_point_bits;
    assert(one == fp_mul(one, one, fixed_point_bits));
    assert(fp_neg(one) == fp_mul(fp_neg(one), one, fixed_point_bits));

    int16_t two = 1 << (fixed_point_bits + 1);
    assert(two == fp_mul(one, two, fixed_point_bits));
    assert(two == fp_mul(two, one, fixed_point_bits));

    int16_t four = 1 << (fixed_point_bits + 2);
    assert(four == fp_mul(two, two, fixed_point_bits));
}

void test_mul_neg(void) {
    int16_t precision = 10;
    
    int16_t one = 1 << precision;
    int16_t negOne = fp_neg(one);
    int16_t prod = fp_mul(one, negOne, precision);
    test_equality(negOne, prod);

    int16_t two = 1 << (precision + 1);
    int16_t negFour = fp_neg(1 << (precision + 2));
    int16_t negEight = fp_neg(1 << (precision + 3));
    prod = fp_mul(two, negFour, precision);
    test_equality(negEight, prod);
}

void test_div_basic(void) {
    int fixed_point_bits = 3;
    int16_t one = 1 << fixed_point_bits;
    assert(one == fp_div(one, one, fixed_point_bits));
    assert(fp_neg(one) == fp_div(fp_neg(one), one, fixed_point_bits));

    int16_t two = 1 << (fixed_point_bits + 1);
    int16_t one_half = 1 << (fixed_point_bits - 1);
    assert(one_half == fp_div(one, two, fixed_point_bits));
    assert(two == fp_div(two, one, fixed_point_bits));

    int16_t four = 1 << (fixed_point_bits + 2);
    assert(two == fp_div(four, two, fixed_point_bits));
    assert(one == fp_div(two, two, fixed_point_bits));
}


void test_mod(void) {
    uint16_t precision = 10;

    int16_t mod = fp_mod(int_to_fp(11, precision), int_to_fp(2, precision), precision);
    assert(int_to_fp(1, precision) == mod);

    mod = fp_mod(int_to_fp(11, precision), int_to_fp(4, precision), precision);
    assert(int_to_fp(3, precision) == mod);

    mod = fp_mod(float_to_fp(7.6, precision), float_to_fp(3.5, precision), precision);
    assert(float_to_fp(0.6, precision) == mod);

    mod = fp_mod(int_to_fp(-11, precision), int_to_fp(4, precision), precision);
    assert(int_to_fp(1, precision) == mod);
}


void test_tanh_basic(void) {
    int fixed_point_bits = 5;
    int16_t zero = 0;
    int16_t one = 1 << fixed_point_bits;
    int16_t two = 1 << (fixed_point_bits + 1);

    assert(0 == fp_tanh(zero, fixed_point_bits));
    assert(24 == fp_tanh(one, fixed_point_bits));
    assert(-24 == fp_tanh(fp_neg(one), fixed_point_bits));
    assert(32 == fp_tanh(two, fixed_point_bits));
    assert(-32 == fp_tanh(fp_neg(two), fixed_point_bits));
}


void test_sigmoid_basic(void) {
    int fixed_point_bits = 8;
    int16_t one_half = 1 << (fixed_point_bits - 1);
    int16_t zero = 0;
    int16_t one = 1 << (fixed_point_bits);
    int16_t two = 1 << (fixed_point_bits + 1);

    assert(one_half == fp_sigmoid(zero, fixed_point_bits));
    assert(186 == fp_sigmoid(one, fixed_point_bits));
    assert(70 == fp_sigmoid(fp_neg(one), fixed_point_bits));
    assert(224 == fp_sigmoid(two, fixed_point_bits));
    assert(32 == fp_sigmoid(fp_neg(two), fixed_point_bits));
}


void test_relu_basic(void) {
    int fixed_point_bits = 8;
    int16_t one_half = 1 << (fixed_point_bits - 1);
    int16_t zero = 0;
    int16_t one = 1 << (fixed_point_bits);
    int16_t two = 1 << (fixed_point_bits + 1);

    assert(zero == fp_relu(zero, fixed_point_bits));
    assert(one == fp_relu(one, fixed_point_bits));
    assert(zero == fp_relu(fp_neg(one), fixed_point_bits));
    assert(two == fp_relu(two, fixed_point_bits));
    assert(zero == fp_relu(fp_neg(two), fixed_point_bits));
}


void test_leaky_relu(void) {
    int16_t precision = 10;
    
    int16_t oneHalf = 1 << (precision - 1);
    int16_t oneFourth = 1 << (precision - 2);
    int16_t oneEighth = 1 << (precision - 3);
    int16_t one = 1 << precision;
    int16_t two = 1 << (precision + 1);

    // Test positive values
    test_equality(two, fp_leaky_relu(two, precision));
    test_equality(one, fp_leaky_relu(one, precision));
    test_equality(oneHalf, fp_leaky_relu(oneHalf, precision));
    test_equality(oneFourth, fp_leaky_relu(oneFourth, precision));
    test_equality(oneEighth, fp_leaky_relu(oneEighth, precision));

    // Test negative values
    test_equality(fp_neg(oneHalf), fp_leaky_relu(fp_neg(two), precision));
    test_equality(fp_neg(oneFourth), fp_leaky_relu(fp_neg(one), precision));
    test_equality(fp_neg(oneEighth), fp_leaky_relu(fp_neg(oneHalf), precision));
}


void test_round_to_int(void) {
    int16_t precision = 4;

    assert(int_to_fp(3, precision) == fp_round_to_int(float_to_fp(3.1, precision), precision));
    assert(int_to_fp(-3, precision) == fp_round_to_int(float_to_fp(-3.1, precision), precision));
    assert(int_to_fp(4, precision) == fp_round_to_int(float_to_fp(3.5, precision), precision));
    assert(int_to_fp(-4, precision) == fp_round_to_int(float_to_fp(-3.5, precision), precision));
}


void test_div32(void) {
    uint16_t precision = 10;

    int32_t x = int_to_fp32(13, precision);
    int32_t y = int_to_fp32(183, precision);
    test_equality(72, fp32_div(x, y, precision));
}


void test_sub32(void) {
    uint16_t precision = 10;

    test_equality(-38156, fp32_sub(71680, 109836));
}


uint8_t test_equality(int32_t expected, int32_t result) {
    if (expected != result) {
        printf("Failed. Expected: %d, Got: %d\n", expected, result);
        return 0;
    }
    return 1;
}


