#include <stdint.h>
#include "math/matrix.h"
#include "layers/cells.h"
#include "layers/layers.h"
#include "utils/neural_network_utils.h"
#include "utils/utils.h"
#include "math/matrix_ops.h"
#include "math/fixed_point_ops.h"
#include "neural_network_parameters.h"

#ifndef NEURAL_NETWORK_GUARD
#define NEURAL_NETWORK_GUARD

struct execution_state {
    uint8_t levelsToExecute;
    uint8_t isStopped;
    int8_t prediction;
    uint8_t isCompleted;
    int16_t cumulativeUpdateProb;  // Used only for Skip RNNs
};
typedef struct execution_state ExecutionState;

uint8_t should_process(uint16_t t, ExecutionState *execState);
void process_input(matrix *input, matrix states[SEQ_LENGTH], matrix logits[NUM_OUTPUTS], uint16_t step, int16_t thresholds[NUM_OUTPUTS], ExecutionState *execState);
matrix *apply_transformation(matrix *result, matrix *input, matrix *state, uint16_t precision);
matrix *compute_logits(matrix *result, matrix *input, uint16_t precision);
int16_t compute_stop_output(matrix *state, uint16_t precision);
matrix *fuse_states(matrix *result, matrix *current, matrix *previous, uint16_t precision);
matrix *pool_states(matrix *result, matrix states[SEQ_LENGTH], dtype weights[SEQ_LENGTH], uint16_t n, uint8_t useSampleStrategy, uint16_t precision);
matrix *pool_logits(matrix *result, matrix logits[NUM_OUTPUTS], matrix states[SEQ_LENGTH], uint16_t n, uint16_t precision);
uint16_t getCurrentLevel(uint16_t seqIndex);
uint8_t isLastSampleInLevel(uint16_t seqIndex, uint16_t fixedSeqIndex);
int16_t get_state_update_prob(matrix *state, int16_t prevUpdateProb, uint16_t precision);
int16_t binarize_update_prob(int16_t updateProb, uint16_t precision);
int16_t phase_gate(int16_t t, uint16_t precision);

#endif
