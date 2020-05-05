#include "main.h"


int main(void) {

    // Initialize a dummy input array
    int16_t numFeatures = 3;
    int16_t seqLength = 4;
    matrix **inputs = (matrix **) alloc(sizeof(matrix *) * seqLength);
    for (int16_t i = 0; i < seqLength; i++) {
        inputs[i] = matrix_allocate(numFeatures, 1);

        for (int j = 0; j < numFeatures; j++) {
            inputs[i]->data[j] = 0;
        }
        
        normalize(inputs[i], INPUT_MEAN, INPUT_STD, FIXED_POINT_PRECISION);
    }

    uint16_t output = execute_model(inputs, seqLength);

    printf("%d\n", output);

    // Release all memory
    for (int16_t i = 0; i < seqLength; i++) {
        matrix_free(inputs[i]);
    }

    free(inputs);
    return 0;
}

