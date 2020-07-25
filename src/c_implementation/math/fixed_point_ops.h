#include <stdint.h>
#include "../utils/utils.h"

#ifndef FIXED_POINT_OPS_GUARD
#define FIXED_POINT_OPS_GUARD

    #define POWER_SERIES_TERMS 7

    int16_t fp_add(int16_t x, int16_t y);
    int16_t fp_mul(int16_t x, int16_t y, uint16_t precision);
    int16_t fp_sub(int16_t x, int16_t y);
    int16_t fp_div(int16_t x, int16_t y, uint16_t precision);
    int16_t fp_neg(int16_t x);
    int16_t fp_exp(int16_t x, uint16_t precision);
    int16_t fp_tanh(int16_t x, uint16_t precision);
    int16_t fp_sigmoid(int16_t x, uint16_t precision);
    int16_t fp_relu(int16_t x, uint16_t precision);
    int16_t fp_leaky_relu(int16_t x, uint16_t precision);
    int16_t fp_linear(int16_t x, uint16_t precision);
    int16_t fp_round_to_int(int16_t x, uint16_t precision);
    int16_t convert_fp(int16_t x, uint16_t old_precision, uint16_t new_precision);
    int16_t float_to_fp(float x, uint16_t precision);
    int16_t int_to_fp(int16_t x, uint16_t precision);

#endif
