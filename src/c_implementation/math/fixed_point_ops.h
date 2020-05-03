#include <stdint.h>

#ifndef FIXED_POINT_OPS_GUARD
#define FIXED_POINT_OPS_GUARD

    #define POWER_SERIES_TERMS 7

    int16_t fp_add(int16_t x, int16_t y);
    int16_t fp_mul(int16_t x, int16_t y, int16_t precision);
    int16_t fp_sub(int16_t x, int16_t y);
    int16_t fp_div(int16_t x, int16_t y, int16_t precision);
    int16_t fp_neg(int16_t x);
    int16_t fp_exp(int16_t x, int16_t precision);
    int16_t fp_tanh(int16_t x, int16_t precision);
    int16_t fp_sigmoid(int16_t x, int16_t precision);
    int16_t convert_fp(int16_t x, int16_t old_precision, int16_t new_precision);

#endif
