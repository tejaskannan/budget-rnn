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


int16_t float_to_fp(float x, int16_t precision) {
    return (int16_t) (x * (1 << precision));
}


int16_t int_to_fp(int16_t x, int16_t precision) {
    return x * (1 << precision);
}


int16_t fp_linear(int16_t x, int16_t precision) {
    UNUSED(precision);
    return x;
}


int16_t fp_exp(int16_t x, int16_t precision) {
    /**
     * Approximates e^x using the Power Series
     */
    int16_t should_invert = 0;
    if (x < 0) {
        x = fp_neg(x);
        should_invert = 1;
    }

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

    if (should_invert) {
        result = fp_div(int_to_fp(1, precision), result, precision);
    }

    return result;
}


int16_t fp_tanh(int16_t x, int16_t precision) {
    /**
     * Approximates tanh using a polynomial.
     */
    int16_t should_invert_sign = 0;
    if (x < 0) {
        x = fp_neg(x);
        should_invert_sign = 1;
    }

    // Create necessary constants
    int16_t twenty_seven = int_to_fp(27, precision);
    int16_t nine = int_to_fp(9, precision);

    int16_t x_squared = fp_mul(x, x, precision);
    int16_t nine_x_squared = fp_mul(nine, x, precision);
    int16_t rational_factor = fp_div(fp_add(twenty_seven, x_squared), fp_add(twenty_seven, nine_x_squared), precision);
    int16_t result = fp_mul(x, rational_factor, precision);

    if (should_invert_sign) {
        result = fp_neg(result);
    }

    int16_t one = int_to_fp(1, precision);
    int16_t neg_one = fp_neg(one);

    if (result > one) {
        return one;    
    } else if (result < neg_one) {
        return neg_one;
    }
    return result;

    // int16_t two = 1 << (precision + 1);
    // int16_t exp2x = fp_exp(fp_mul(x, two, precision), precision);
 //   int16_t exp = fp_exp(x, precision);
 //   int16_t exp_neg = fp_exp(fp_neg(x), precision);

 //   if (fp_add(exp, exp_neg) == 0) {
 //       exp = fp_exp(x, precision);
 //   }

 //   int16_t result = fp_div(fp_sub(exp, exp_neg), fp_add(exp, exp_neg), precision);

 //   if (should_invert_sign)
 //       return fp_neg(result);
 //   return result;
}


int16_t fp_sigmoid(int16_t x, int16_t precision) {
    /**
     * Approximates the sigmoid function using absolute values.
     */
    int16_t one = 1 << precision;
    int16_t one_half = 1 << (precision - 1);

    int16_t absX = x;
    if (x < 0) {
        absX = fp_neg(x);
    }

    int16_t rational_factor = fp_div(x, fp_add(absX, one), precision);
    int16_t result = fp_add(fp_mul(rational_factor, one_half, precision), one_half);
    return result;

  //  int16_t exp_neg_x = fp_exp(fp_neg(x), precision);
  //  
  //  int16_t exp_neg_x_plus_one = fp_add(exp_neg_x, one);
  //  if (exp_neg_x_plus_one == 0) {
  //      exp_neg_x_plus_one = fp_add(exp_neg_x, one);
  //  }
  //  
  //  int16_t result = fp_div(one, exp_neg_x_plus_one, precision);

  //  if (should_invert_sign)
  //      return fp_neg(result);
  //  return result;
}