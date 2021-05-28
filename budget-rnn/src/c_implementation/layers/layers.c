#include "layers.h"


matrix *dense(matrix *result, matrix *input, matrix *W, matrix *b, int16_t (*activation)(int16_t, uint16_t), uint16_t precision) {
    /**
     * Implementation of a dense feed-forward layer using matrix operations.
     */
    result = matrix_multiply(result, W, input, precision);

    // Only add bias if given 
    if (b != NULL_PTR) {
        result = matrix_add(result, result, b);
    }

    result = apply_elementwise(result, result, activation, precision);
    return result;
}


matrix *apply_gate(matrix *result, matrix *gate, matrix *first, matrix *second, matrix *temp, uint16_t precision) {
    // Create the vector for 1 - gate
    temp = matrix_neg(temp, gate, precision);
    temp = scalar_add(temp, temp, int_to_fp(1, precision));

    // temp = (1.0 - gate) * second
    temp = matrix_hadamard(temp, second, temp, precision);

    // result = gate * first
    result = matrix_hadamard(result, first, gate, precision);

    // result += temp -> result = gate * first + (1.0 - gate) * second
    result = matrix_add(result, result, temp);

    return result;
}


matrix *apply_gru(matrix *result, matrix *input, matrix *state, GRU *gru, GRUTempStates *temp, uint16_t precision) {
    /**
     * Implementation of a GRU Cell.
     */
    // Unpack memory for intermediate states
    matrix *gates = temp->gates;
    matrix *candidate = temp->candidate;
    matrix *gateTemp = temp->gateTemp;
    matrix *stacked = temp->stacked;

    // Concatenate state and inputs for combined matrix multiplication
    matrix *state_input_concat = vstack(stacked, state, input);

    // Create the gates
    gates = matrix_multiply(gates, gru->wGates, state_input_concat, precision);
    gates = matrix_add(gates, gates, gru->bGates);
    gates = apply_elementwise(gates, gates, &fp_sigmoid, precision);

    // Split gates into reset and update components
    matrix update = { gates->data, STATE_SIZE, gates->numCols };

    uint16_t offset = update.numRows * update.numCols;
    matrix reset = { gates->data + offset, STATE_SIZE, gates->numCols };
    matrix_hadamard(&reset, state, &reset, precision);

    // Concatenate state and inputs to account for reset gate
    matrix *reset_input_concat = vstack(stacked, &reset, input);

    // Create the candidate state
    candidate = matrix_multiply(candidate, gru->wCandidate, reset_input_concat, precision);
    candidate = matrix_add(candidate, candidate, gru->bCandidate);
    candidate = apply_elementwise(candidate, candidate, &fp_tanh, precision);

    // Construct the result
    result = apply_gate(result, &update, state, candidate, gateTemp, precision);
 
    return result;
}


matrix *apply_ugrnn(matrix *result, matrix *input, matrix *state, UGRNN *gru, UGRNNTempStates *temp, uint16_t precision) {
    /**
     * Implementation of a UGRNN Cell. This cell has a single update gate.
     */
    // Unpack memory for intermediate states
    matrix *stacked = temp->stacked;
    matrix *transformed = temp->transformed;
    matrix *gateTemp = temp->gateTemp;

    // Concatenate state and inputs
    matrix *state_input_concat = vstack(stacked, state, input);

    // Compute the transformation
    transformed = matrix_multiply(transformed, gru->wTransform, state_input_concat, precision);
    transformed = matrix_add(transformed, transformed, gru->bTransform);
    
    // Split transformation into update and candidate components
    matrix update = { transformed->data, STATE_SIZE, transformed->numCols };

    uint16_t offset = update.numRows * update.numCols;
    matrix candidate = { transformed->data + offset, STATE_SIZE, transformed->numCols };

    // Create the update gate
    scalar_add(&update, &update, int_to_fp(1, precision));  // Add initial bias term
    apply_elementwise(&update, &update, &fp_sigmoid, precision);

    // Create the candidate state
    apply_elementwise(&candidate, &candidate, &fp_tanh, precision);

    // Construct the result
    result = apply_gate(result, &update, state, &candidate, gateTemp, precision);
 
    return result;
}
