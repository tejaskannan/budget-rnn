#include <stdint.h>
#include "../neural_network_parameters.h"
#include "../math/fixed_point_ops.h"

#ifndef INTERPOLATION_GUARD
#define INTERPOLATION_GUARD

    struct InterpolationResult {
        int16_t weight;
        int16_t lowerIdx;
        int16_t upperIdx;
    };
    typedef struct InterpolationResult InterpolationResult;

    InterpolationResult get_interpolation_values(int16_t target, uint16_t precision);
    int16_t *interpolate_thresholds(int16_t *result, int16_t target, uint16_t precision);
    void interpolate_counts(int32_t result[NUM_OUTPUT_FEATURES][NUM_OUTPUTS], int16_t target, uint16_t precision);

#endif
