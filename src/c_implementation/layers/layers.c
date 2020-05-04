#include "layers.h"


matrix *dense(matrix *result, matrix *input, matrix *W, matrix *b, int16_t (*activation)(int16_t, int16_t), int16_t precision) {
    /**
     * Implementation of a dense feed-forward layer using matrix operations.
     */
    result = matrix_multiply(result, input, W, precision);

    if (!isNull(b)) {
        result = matrix_add(result, result, b);
    }

    result = apply_elementwise(result, result, activation, precision);
    return result;
}


matrix *apply_gate(matrix *result, matrix *gate, matrix *first, matrix *second, int16_t precision) {
    // Create the vector for 1 - gate
    matrix *opp_gate = matrix_allocate(gate->numRows, gate->numCols);
    opp_gate = scalar_add(opp_gate, scalar_product(opp_gate, opp_gate, int_to_fp(-1, precision), precision), int_to_fp(1, precision));
    
    opp_gate = matrix_hadamard(opp_gate, second, opp_gate, precision);
    result = matrix_add(result, matrix_hadamard(result, first, gate, precision), opp_gate);

    matrix_free(opp_gate);
    return result;
}


matrix *apply_gru(matrix *result, matrix *input, matrix *state, GRU *gru, int16_t precision) {
    /**
     * Implementation of a GRU Cell.
     */
    // Allocate matrices for the intermediate state
    matrix *update = matrix_allocate(state->numRows, state->numCols);
    matrix *reset = matrix_allocate(state->numRows, state->numCols);
    matrix *candidate = matrix_allocate(state->numRows, state->numCols);

    // Create the update state
    update = matrix_multiply(update, gru->wUpdate, state, precision);
    update = matrix_add(update, update, matrix_multiply(update, gru->uUpdate, input, precision));
    update = matrix_add(update, update, gru->bUpdate);
    update = apply_elementwise(update, update, &fp_sigmoid, precision);

    // Create the reset state
    reset = matrix_multiply(reset, gru->wReset, state, precision);
    reset = matrix_add(reset, reset, matrix_multiply(reset, gru->uReset, input, precision));
    reset = matrix_add(reset, reset, gru->bReset);
    reset = apply_elementwise(reset, reset, &fp_sigmoid, precision);

    // Create the candidate state
    reset = matrix_hadamard(reset, state, reset, precision);
    candidate = matrix_multiply(candidate, gru->wCandidate, reset, precision);
    candidate = matrix_add(candidate, candidate, matrix_multiply(candidate, gru->uCandidate, input, precision));
    candidate = matrix_add(candidate, candidate, gru->bCandidate);
    candidate = apply_elementwise(candidate, candidate, &fp_tanh, precision);

    // Construct the result
    result = apply_gate(result, update, candidate, state, precision);
 
    // Free intermediate states
    matrix_free(update);
    matrix_free(reset);
    matrix_free(candidate);
 
    return result;
}


matrix *rnn(matrix *result, matrix **inputs, void *cell, enum CellType cellType, int16_t seqLength, int16_t precision) {
    /**
     * Implementation of an RNN that outputs the final state to summarize the input sequence.
     */
    // The output is the final state
    matrix *state = result;
    matrix_set(state, 0);  // Start with a zero state.

    int16_t i;
    for (i = 0; i < seqLength; i++) {
        matrix *input = inputs[i];

        if (cellType == GRUCell) {
            state = apply_gru(state, input, state, ((GRU *) cell), precision);
        } else {
            return NULL_PTR;
        }
    }

    return state;
}



