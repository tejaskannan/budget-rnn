#include "fixed_point_ops.h"


int16_t fp_add(int16_t x, int16_t y) {
    return x + y;
}


int16_t fp_sub(int16_t x, int16_t y) {
    return x - y;
}


int16_t fp_mul(int16_t x, int16_t y, int16_t precision) {
    return (int16_t) (((int) x * (int) y) / (1 << precision));
}


int16_t fp_div(int16_t x, int16_t y, int16_t precision) {
    return (int16_t) (((int) x * (1 << precision)) / y);
}


int16_t fp_neg(int16_t x) {
    return -1 * x;
}


int16_t convert_fp(int16_t x, int16_t old_precision, int16_t new_precision) {
    return (x * (1 << new_precision)) / (1 << old_precision);
}


int16_t fp_exp(int16_t x, int16_t precision) {
    /**
     * Approximates e^x using the Power Series
     */
    int16_t result = 1 << precision;
    int16_t prev_result = 0;

    int16_t acc = 1 << precision;
    int16_t fact = 1 << precision;
    int16_t term;
    int i;
    for (i = 1; i < POWER_SERIES_TERMS && prev_result != result; i++) {
        acc = fp_mul(x, acc, precision);

        int16_t factor = (int16_t) (i * (1 << precision));
        fact = fp_mul(fact, factor, precision);

        term = fp_div(acc, fact, precision);

        prev_result = result;
        result = fp_add(term, result);
    }

    return result;
}


int16_t fp_tanh(int16_t x, int16_t precision) {
    /**
     * Approximates tanh using e^x.
     */
    int16_t should_invert_sign = 0;
    if (x < 0) {
        x = fp_neg(x);
        should_invert_sign = 1;
    }

    int16_t two = 1 << (precision + 1);
    int16_t exp2x = fp_exp(fp_mul(x, two, precision), precision);

    int16_t one = 1 << precision;
    int16_t neg_one = fp_neg(one);

    int16_t result = fp_div(fp_add(exp2x, neg_one), fp_add(exp2x, one), precision);

    if (should_invert_sign)
        return fp_neg(result);
    return result;
}


int16_t fp_sigmoid(int16_t x, int16_t precision) {
    /**
     * Approximates the sigmoid function using e^x
     */
    int16_t should_invert_sign = 0;
    if (x < 0) {
        x = fp_neg(x);
        should_invert_sign = 1;
    }

    int16_t one = 1 << precision;
    int16_t exp_neg_x = fp_exp(fp_neg(x), precision);
    int16_t result = fp_div(one, fp_add(one, exp_neg_x), precision);

    if (should_invert_sign)
        return fp_neg(result);
    return result;
}
