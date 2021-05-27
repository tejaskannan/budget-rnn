#include "interpolation.h"

InterpolationResult get_interpolation_values(int16_t target, uint16_t precision) {
    int16_t lowerIdx = -1;
    int16_t upperIdx = NUM_BUDGETS;

    uint16_t i = NUM_BUDGETS;
    for (; i > 1; i--) {
        if (target == BUDGETS[i-1]) {
            lowerIdx = i-1;
            upperIdx = i-1;
            break;
        } else if (target > BUDGETS[i-2] && target < BUDGETS[i-1]) {
            lowerIdx = i-2;
            upperIdx = i-1;
            break;
        }
    }

    int16_t one = 1 << precision;
    int16_t weight = one;

    if (target == BUDGETS[0]) {
        lowerIdx = 0;
        upperIdx = 0;
    } else if (target < BUDGETS[0]) {
        lowerIdx = -1;
        upperIdx = -1;
    } else if (target > BUDGETS[NUM_BUDGETS - 1]) {
        lowerIdx = NUM_BUDGETS;
        upperIdx = NUM_BUDGETS;
    } else if (lowerIdx < upperIdx) {
        int16_t num = fp_sub(target, AVG_ENERGY[lowerIdx]);
        int16_t denom = fp_sub(AVG_ENERGY[upperIdx], AVG_ENERGY[lowerIdx]);
        weight = fp_div(num, denom, precision);
    }

    if (weight < 0) {
        weight = 0;
    } else if (weight > one) {
        weight = one;
    }
    
    InterpolationResult result;
    result.lowerIdx = lowerIdx;
    result.upperIdx = upperIdx;
    result.weight = weight;

    return result;
}


int16_t *interpolate_thresholds(int16_t *result, int16_t target, uint16_t precision) {
    InterpolationResult values = get_interpolation_values(target, precision);
    uint16_t i;

    if (values.lowerIdx < 0) {
        for (i = NUM_OUTPUTS; i > 0; i--) {
            result[i-1] = 0;
        }
    } else if (values.upperIdx >= NUM_BUDGETS) {
        for (i = NUM_OUTPUTS; i > 0; i--) {
            result[i-1] = 1 << precision;
        }
    } else {
        int16_t *lowerThresh = THRESHOLDS[(uint16_t) values.lowerIdx];
        int16_t *upperThresh = THRESHOLDS[(uint16_t) values.upperIdx];
        int16_t lower, upper;

        for (i = NUM_OUTPUTS; i > 0; i--) {
            lower = fp_mul(lowerThresh[i-1], fp_sub(1 << precision, values.weight), precision);
            upper = fp_mul(upperThresh[i-1], values.weight, precision);
            result[i-1] = fp_add(lower, upper);
        }
    }

    result[NUM_OUTPUTS - 1] = 0;

    return result;
}


void interpolate_counts(int32_t result[NUM_OUTPUT_FEATURES][NUM_OUTPUTS], int16_t target, uint16_t precision) {
    InterpolationResult values = get_interpolation_values(target, precision);

    int16_t lowerIdx = values.lowerIdx;
    if (lowerIdx < 0) {
        lowerIdx = 0;
    } else if (lowerIdx >= NUM_BUDGETS) {
        lowerIdx = NUM_BUDGETS - 1;
    }

    int16_t upperIdx = values.upperIdx;
    if (upperIdx < 0) {
        upperIdx = 0;
    } else if (upperIdx >= NUM_BUDGETS) {
        upperIdx = NUM_BUDGETS - 1;
    }

    int32_t lower, upper;
    uint16_t i, j;

    for (i = NUM_OUTPUT_FEATURES; i > 0; i--) {
        for (j = NUM_OUTPUTS; j > 0; j--) {
            lower = fp32_mul(int_to_fp32(LABEL_COUNTS[lowerIdx][i-1][j-1], precision), fp32_sub(1 << precision, values.weight), precision);
            upper = fp32_mul(int_to_fp32(LABEL_COUNTS[upperIdx][i-1][j-1], precision), values.weight, precision);
            result[i-1][j-1] = fp32_add(lower, upper);
        }
    }
}
