#include <stdint.h>

#ifndef NN_UTILS_GUARD
#define NN_UTILS_GUARD

struct InferenceResult {
    int16_t prediction;
    uint8_t numLevels;
    uint8_t hasStoppedEarly;
};
typedef struct InferenceResult InferenceResult;

#endif
