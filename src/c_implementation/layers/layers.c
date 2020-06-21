#include "layers.h"


matrix *dense(matrix *result, matrix *input, matrix *W, matrix *b, int16_t (*activation)(int16_t, int16_t), uint8_t is_compressed, char *seed, int16_t precision) {
    /**
     * Implementation of a dense feed-forward layer using matrix operations.
     */

    // If the weights are compressed, then we use the hashing trick to compute the matrix multiplication. Otherwise,
    // we  perform standard matrix multiplication.
    if (is_compressed) {
        result = hashed_matrix_vector_product(result, W, input, seed, 1, precision);
    } else { 
        result = matrix_multiply(result, W, input, precision);
    }

    if (!isNull(b)) {
        result = matrix_add(result, result, b);
    }

    result = apply_elementwise(result, result, activation, precision);
    return result;
}


matrix *apply_gate(matrix *result, matrix *gate, matrix *first, matrix *second, matrix *temp, int16_t precision) {
    // Create the vector for 1 - gate
    temp = scalar_product(temp, gate, int_to_fp(-1, precision), precision);
    temp = scalar_add(temp, temp, int_to_fp(1, precision));

    temp = matrix_hadamard(temp, second, temp, precision);
    result = matrix_add(result, matrix_hadamard(result, first, gate, precision), temp);

    return result;
}


matrix *apply_gru(matrix *result, matrix *input, matrix *state, GRU *gru, GRUTempStates *temp, uint8_t is_compressed, uint8_t layer, int16_t precision) {
    /**
     * Implementation of a GRU Cell.
     */
    // Unpack memory for intermediate states
    matrix *update = temp->update;
    matrix *reset = temp->reset;
    matrix *candidate = temp->candidate;
    matrix *inputTemp = temp->inputTemp;
    matrix *tempGate = temp->gateTemp;

    char hash_seed[7];
    replace(hash_seed, TRANSFORM_SEED, 0);
    hash_seed[5] = (char) (layer + '0');
    hash_seed[6] = '\0'; // Make sure the seed is null-terminated

    // Create the update state
    if (is_compressed) {
        hash_seed[2] = 'U';
        replace(hash_seed, UPDATE_SEED, 3);
        inputTemp = hashed_matrix_vector_product(inputTemp, gru->uUpdate, input, hash_seed, 1, precision);

        hash_seed[2] = 'W';
        update = hashed_matrix_vector_product(update, gru->wUpdate, state, hash_seed, 1, precision);
    } else {
        inputTemp = matrix_multiply(inputTemp, gru->uUpdate, input, precision);
        update = matrix_multiply(update, gru->wUpdate, state, precision);
    }

    update = matrix_add(update, update, inputTemp);
    update = matrix_add(update, update, gru->bUpdate);
    update = apply_elementwise(update, update, &fp_sigmoid, precision);

    // Create the reset state
    if (is_compressed) {
        hash_seed[2] = 'U';
        replace(hash_seed, RESET_SEED, 3);
        inputTemp = hashed_matrix_vector_product(inputTemp, gru->uReset, input, hash_seed, 1, precision);
        
        hash_seed[2] = 'W';
        reset = hashed_matrix_vector_product(reset, gru->wReset, state, hash_seed, 1, precision);
    } else {
        inputTemp = matrix_multiply(inputTemp, gru->uReset, input, precision);
        reset = matrix_multiply(reset, gru->wReset, state, precision);
    }

    reset = matrix_add(reset, reset, inputTemp);
    reset = matrix_add(reset, reset, gru->bReset);
    reset = apply_elementwise(reset, reset, &fp_sigmoid, precision);
    reset = matrix_hadamard(reset, state, reset, precision);

    // Create the candidate state
    if (is_compressed) {
        hash_seed[2] = 'U';
        replace(hash_seed, CANDIDATE_SEED, 3);
        inputTemp = hashed_matrix_vector_product(inputTemp, gru->uCandidate, input, hash_seed, 1, precision);

        hash_seed[2] = 'W';
        candidate = hashed_matrix_vector_product(candidate, gru->wCandidate, reset, hash_seed, 1, precision);
    } else {
        inputTemp = matrix_multiply(inputTemp, gru->uCandidate, input, precision);
        candidate = matrix_multiply(candidate, gru->wCandidate, reset, precision);
    }
    
    candidate = matrix_add(candidate, candidate, inputTemp);
    candidate = matrix_add(candidate, candidate, gru->bCandidate);
    candidate = apply_elementwise(candidate, candidate, &fp_tanh, precision);

    // Construct the result
    result = apply_gate(result, update, state, candidate, tempGate, precision);
 
    return result;
}

matrix *apply_tf_gru(matrix *result, matrix *input, matrix *state, TFGRU *gru, TFGRUTempStates *tempStates, int16_t precision) {
    /**
     * Implementation of a Tensorflow GRU Cell. This implementation applies a matrix to a stack version of state
     * and inputs instead of separating the matrices out.
     */
    // Allocate matrices for the intermediate state
   // matrix *stacked = matrix_allocate(input->numRows + state->numRows, state->numCols);
   // matrix *gates = matrix_allocate(state->numRows * 2, state->numCols);
   // matrix *candidate = matrix_allocate(state->numRows, state->numCols);
   // matrix *update = matrix_allocate(state->numRows, state->numCols);
   // matrix *reset = matrix_allocate(state->numRows, state->numCols);
   // matrix *tempGate = matrix_allocate(state->numRows, state->numCols);

    // Unpack the temp states
    matrix *stacked = tempStates->stacked;
    matrix *gates = tempStates->gates;
    matrix *candidate = tempStates->candidate;
    matrix *update = tempStates->update;
    matrix *reset = tempStates->reset;
    matrix *tempGate = tempStates->tempGate;
    // matrix *tempGate = matrix_allocate(state->numRows, state->numCols);

    // Create the gates
    stacked = stack(stacked, input, state);
    gates = matrix_multiply(gates, gru->wGates, stacked, precision);
    gates = matrix_add(gates, gates, gru->bGates);
    gates = apply_elementwise(gates, gates, &fp_sigmoid, precision);

    // Split the gates into reset and update components
    int16_t index = 0;
    for (; index < state->numRows; index++) {
        reset->data[index] = gates->data[index];
    }

    int16_t offset = index;
    for (; index < gates->numRows; index++) {
        update->data[index - offset] = gates->data[index];
    }

    // Create the candidate state
    reset = matrix_hadamard(reset, state, reset, precision);
    stacked = stack(stacked, input, reset);

    candidate = matrix_multiply(candidate, gru->wCandidates, stacked, precision);
    candidate = matrix_add(candidate, candidate, gru->bCandidates);
    candidate = apply_elementwise(candidate, candidate, &fp_tanh, precision);

    // Construct the result
    result = apply_gate(result, update, state, candidate, tempGate, precision);

    // Free intermediate states
//    matrix_free(update);
//    matrix_free(reset);
//    matrix_free(candidate);
//    matrix_free(stacked);
//    matrix_free(gates);
 
    return result;
}
