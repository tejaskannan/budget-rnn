#include "neural_network.h"

// Buffer for intermediate states
static dtype DATA_BUFFER[7 * STATE_SIZE * VECTOR_COLS] = {0};
static dtype STATE_BUFFER[3 * STATE_SIZE * VECTOR_COLS + NUM_OUTPUT_FEATURES * VECTOR_COLS] = {0};
static dtype INPUT_BUFFER[NUM_INPUT_FEATURES * VECTOR_COLS] = {0};


uint8_t should_process(uint16_t t, ExecutionState *execState) {
    #if defined(IS_BUDGET_RNN)
        uint16_t currentLevel  = getCurrentLevel(t);
        return currentLevel <= execState->levelsToExecute || !execState->isStopped;
    #elif defined(IS_SKIP_RNN)
        UNUSED(t);
        int16_t binaryUpdateProb = binarize_update_prob(execState->cumulativeUpdateProb, FIXED_POINT_PRECISION);
        return (uint8_t) (binaryUpdateProb > 0);
    #elif defined(IS_PHASED_RNN)
        UNUSED(execState);
        int16_t gate = phase_gate(t, FIXED_POINT_PRECISION);
        return (uint8_t) (gate > 0);
    #else
        return (uint8_t) (t <= execState->levelsToExecute);
    #endif
}


void process_input(matrix *input, matrix states[SEQ_LENGTH], matrix logits[NUM_OUTPUTS], uint16_t step, int16_t thresholds[NUM_OUTPUTS], ExecutionState *execState) {
    /**
     * Processes the current input using the recurrent neural network. This function tracks when inference should stop
     * through the execution state struct.
     */
    #ifndef IS_BUDGET_RNN
    UNUSED(thresholds);
    UNUSED(logits);
    #endif

    uint16_t currentLevel = getCurrentLevel(step);

    uint16_t stateBufferOffset = 0;

    matrix currentState = { STATE_BUFFER, STATE_SIZE, VECTOR_COLS };
    stateBufferOffset += currentState.numRows * currentState.numCols;

    if (step == 0) {
        matrix_set(&currentState, 0);
    } else {
    #if defined(IS_BUDGET_RNN) && STRIDE_LENGTH > 1
        // Set state from the last sample in the current sub-sequence
        matrix prevSampleState = { STATE_BUFFER + stateBufferOffset, STATE_SIZE, VECTOR_COLS };
        stateBufferOffset += prevSampleState.numRows * prevSampleState.numCols;

        if (step >= NUM_OUTPUTS) {
            matrix_replace(&prevSampleState, states + step - NUM_OUTPUTS);
        } else {
            matrix_set(&prevSampleState, 0);
        }

        // The first level does not use a fusion layer
        if (currentLevel == 0) {
            matrix_replace(&currentState, &prevSampleState);
        } else {
            matrix prevLevelState = { STATE_BUFFER + stateBufferOffset, STATE_SIZE, VECTOR_COLS };
            matrix_replace(&prevLevelState, states + step - 1);

            fuse_states(&currentState, &prevSampleState, &prevLevelState, FIXED_POINT_PRECISION);
        }

        // Reset the state buffer offset to `free` memory
        stateBufferOffset -= prevSampleState.numRows * prevSampleState.numCols;
    #else
        matrix_replace(&currentState, states + step - 1);
    #endif
    }

    // Apply recurrent cell
    matrix nextState = { STATE_BUFFER + stateBufferOffset, STATE_SIZE, VECTOR_COLS };
    stateBufferOffset += nextState.numRows * nextState.numCols;

    matrix inputFeatures = { INPUT_BUFFER, NUM_INPUT_FEATURES, VECTOR_COLS };
    matrix_replace(&inputFeatures, input);

    apply_transformation(&nextState, &inputFeatures, &currentState, FIXED_POINT_PRECISION);

    #ifdef IS_BUDGET_RNN
    #if STRIDE_LENGTH > 1
    if (step < NUM_OUTPUTS) {
    #else
    if ((step + 1) % SAMPLES_PER_SEQ == 0) {
    #endif
        int16_t stopProb = compute_stop_output(&nextState, FIXED_POINT_PRECISION);
        int16_t threshold = thresholds[currentLevel];

        if (threshold < stopProb || currentLevel == NUM_OUTPUTS) {
            execState->levelsToExecute = currentLevel;
            execState->isStopped = 1;
        }
    }
    #endif

    #ifdef IS_PHASED_RNN
    // Apply the phase gate
    int16_t gate = phase_gate(step, FIXED_POINT_PRECISION);
    int16_t oneMinusGate = fp_add(int_to_fp(1, FIXED_POINT_PRECISION), fp_neg(gate));

    matrix prevState = { STATE_BUFFER + stateBufferOffset, STATE_SIZE, VECTOR_COLS };

    scalar_product(&nextState, &nextState, gate, FIXED_POINT_PRECISION);
    scalar_product(&prevState, &currentState, oneMinusGate, FIXED_POINT_PRECISION);
    matrix_add(&nextState, &nextState, &prevState);
    #endif

    // Save next state into the states array
    matrix_replace(states + step, &nextState);

    #if defined(IS_BUDGET_RNN)
    // If we reach the last element of the highest level, we compute the output.
    uint8_t isEnd = isLastSampleInLevel(step, execState->levelsToExecute);

    if (isEnd) {
        compute_logits(logits + currentLevel, &nextState, FIXED_POINT_PRECISION);
    }

    if ((currentLevel == execState->levelsToExecute && execState->isStopped && isEnd) || (step == SEQ_LENGTH - 1)) {
        matrix pooledLogits = { STATE_BUFFER + stateBufferOffset, NUM_OUTPUT_FEATURES, VECTOR_COLS };
        pool_logits(&pooledLogits, logits, states, currentLevel + 1, FIXED_POINT_PRECISION);
        execState->prediction = argmax(&pooledLogits);
        execState->isCompleted = 1;
    }
    #elif defined(IS_RNN)
    uint8_t isEnd = isLastSampleInLevel(step, execState->levelsToExecute);

    if (isEnd) {
        matrix logProbs = { STATE_BUFFER + stateBufferOffset, NUM_OUTPUT_FEATURES, VECTOR_COLS };
        compute_logits(&logProbs, &nextState, FIXED_POINT_PRECISION);
        execState->prediction = argmax(&logProbs);
        execState->isCompleted = 1;
    }
    #else
    matrix logProbs = { STATE_BUFFER + stateBufferOffset, NUM_OUTPUT_FEATURES, VECTOR_COLS };
    compute_logits(&logProbs, &nextState, FIXED_POINT_PRECISION);
    execState->prediction = argmax(&logProbs);
    #endif
}


uint16_t getCurrentLevel(uint16_t seqIndex) {
    #ifdef IS_BUDGET_RNN
    if (STRIDE_LENGTH > 1) {
        return seqIndex % NUM_OUTPUTS;
    } else {
        return seqIndex / SAMPLES_PER_SEQ;
    }
    #else
    return seqIndex;
    #endif
}


uint8_t isLastSampleInLevel(uint16_t seqIndex, uint16_t fixedSeqIndex) {
    #ifdef IS_BUDGET_RNN
    UNUSED(fixedSeqIndex);
    if (STRIDE_LENGTH > 1) {
        return (SEQ_LENGTH - seqIndex) <= NUM_OUTPUTS;
    } else {
        return ((seqIndex + 1) % SAMPLES_PER_SEQ) == 0;
    }
    #else
    return seqIndex == fixedSeqIndex;
    #endif
}


matrix *apply_transformation(matrix *result, matrix *input, matrix *state, uint16_t precision) {
    /**
     * Applies both the embedding layer and the transformation to the input and previous state.
     */
    uint16_t data_buffer_offset = 0;

    // Allocate intermediate state for the embedding
    matrix embedding = { DATA_BUFFER + data_buffer_offset, state->numRows, state->numCols };
    data_buffer_offset += embedding.numRows * embedding.numCols;

    // Apply the embedding layer
    dense(&embedding, input, EMBEDDING_LAYER_KERNEL_MAT, EMBEDDING_LAYER_BIAS_MAT, &fp_leaky_relu, precision);

    // Apply the specified transformation layer
    #ifdef GRU_TRANSFORM
        // Allocate temporary states
        matrix stacked = { DATA_BUFFER + data_buffer_offset, 2 * state->numRows, state->numCols };
        data_buffer_offset += stacked.numRows * stacked.numCols;

        matrix gates = { DATA_BUFFER + data_buffer_offset, 2 * state->numRows, state->numCols };
        data_buffer_offset += gates.numRows * gates.numCols;

        matrix candidate = { DATA_BUFFER + data_buffer_offset, state->numRows, state->numCols };
        data_buffer_offset += candidate.numRows * candidate.numCols;

        matrix gateTemp = { DATA_BUFFER + data_buffer_offset, state->numRows, state->numCols };
        data_buffer_offset += gateTemp.numRows * gateTemp.numCols;

        GRUTempStates rnnTemp;
        rnnTemp.stacked = &stacked;
        rnnTemp.gates = &gates;
        rnnTemp.candidate = &candidate;
        rnnTemp.gateTemp = &gateTemp;

        /// Create the GRU Cell
	    GRU rnn_cell = { RNN_CELL_W_GATES_MAT, RNN_CELL_B_GATES_MAT, RNN_CELL_W_CANDIATE_MAT, RNN_CELL_B_CANDIDATE_MAT };

        // Apply the GRU Cell
        apply_gru(result, &embedding, state, &rnn_cell, &rnnTemp, precision);
    #elif defined(UGRNN_TRANSFORM)
        // Allocate temporary states
        matrix stacked = { DATA_BUFFER + data_buffer_offset, 2 * state->numRows, state->numCols };
        data_buffer_offset += stacked.numRows * stacked.numCols;

        matrix transformed = { DATA_BUFFER + data_buffer_offset, 2 * state->numRows, state->numCols };
        data_buffer_offset += transformed.numRows * transformed.numCols;

        matrix gateTemp = { DATA_BUFFER + data_buffer_offset, state->numRows, state->numCols };
        data_buffer_offset += gateTemp.numRows * gateTemp.numCols;

        UGRNNTempStates rnnTemp;
        rnnTemp.stacked = &stacked;
        rnnTemp.transformed = &transformed;
        rnnTemp.gateTemp = &gateTemp;

        /// Create the UGRNN Cell
	    UGRNN rnn_cell = { RNN_CELL_W_TRANSFORM_MAT, RNN_CELL_B_TRANSFORM_MAT };

        // Apply the GRU Cell
        apply_ugrnn(result, &embedding, state, &rnn_cell, &rnnTemp, precision);
    #endif

    return result;
}


// Function to compute output (1 or 2 hidden layer depending on model type)
matrix *compute_logits(matrix *result, matrix *input, uint16_t precision) {
    /**
     * Function to compute the prediction using a feed-forward network.
     */
    // Allocate intermediate states
    uint16_t data_buffer_offset = 0;
    matrix hidden = { DATA_BUFFER + data_buffer_offset, OUTPUT_LAYER_HIDDEN_0_BIAS_MAT->numRows, input->numCols };
    data_buffer_offset += hidden.numRows * hidden.numCols;

    // Apply the dense layers
    dense(&hidden, input, OUTPUT_LAYER_HIDDEN_0_KERNEL_MAT, OUTPUT_LAYER_HIDDEN_0_BIAS_MAT, &fp_leaky_relu, precision);
    dense(result, &hidden, OUTPUT_LAYER_OUTPUT_KERNEL_MAT, OUTPUT_LAYER_OUTPUT_BIAS_MAT, &fp_linear, precision);

    return result;
}


#ifdef IS_BUDGET_RNN
int16_t compute_stop_output(matrix *state, uint16_t precision) {
    /**
     * Computes the stop output from the given state.
     */
    // Allocate intermediate states
    uint16_t data_buffer_offset = 0;
    matrix hidden = { DATA_BUFFER + data_buffer_offset, STOP_PREDICTION_HIDDEN_0_BIAS_MAT->numRows, state->numCols };
    data_buffer_offset += hidden.numRows * hidden.numCols;

    // Apply the hidden layer
    dense(&hidden, state, STOP_PREDICTION_HIDDEN_0_KERNEL_MAT, STOP_PREDICTION_HIDDEN_0_BIAS_MAT, &fp_leaky_relu, precision);

    // Apply the output layer. We do this manually to prevent the matrix multiplication from engaging the LEA. The stop output
    // is a single value, and the LEA can only operate on matrices with even dimensions.
    int16_t output = dot_product(STOP_PREDICTION_OUTPUT_KERNEL_MAT, &hidden, precision);
    output = fp_add(output, STOP_PREDICTION_OUTPUT_BIAS_MAT->data[0]);
    output = fp_sigmoid(output, precision);

    return output;
}

matrix *pool_logits(matrix *result, matrix logits[NUM_OUTPUTS], matrix states[SEQ_LENGTH], uint16_t n, uint16_t precision) {
    if (n <= 1) {
        matrix_replace(result, &logits[0]);
        return result;
    }

    // Set the result data to zero.
    matrix_set(result, 0);

    // Compute the aggregation weights using the model parameters
    uint16_t dataBufferOffset = 0;
    matrix aggregationWeights = { DATA_BUFFER + dataBufferOffset, n, VECTOR_COLS };
    dataBufferOffset += aggregationWeights.numRows * aggregationWeights.numCols;

    matrix stacked = { DATA_BUFFER + dataBufferOffset, 2 * STATE_SIZE, VECTOR_COLS };
    dataBufferOffset += stacked.numRows * stacked.numCols;

    int16_t weight;
    uint16_t offset;

    #if STRIDE_LENGTH > 1
    offset = SEQ_LENGTH - NUM_OUTPUTS + n - 1;
    #else
    offset = n * SAMPLES_PER_SEQ - 1;
    #endif

    matrix *anchor = states + offset;

    uint16_t i;
    matrix *state;
    for (i = n; i > 0; i--) {
        // Get (i-1)-th final state depending on the stride length
        #if STRIDE_LENGTH > 1
        offset = SEQ_LENGTH - NUM_OUTPUTS + i - 1;
        #else
        offset = i * SAMPLES_PER_SEQ - 1;
        #endif

        // Stack the states
        state = states + offset;
        vstack(&stacked, state, anchor);

        // Apply the aggregation layer
        weight = dot_product(AGGREGATION_LAYER_KERNEL_MAT, &stacked, precision);
        weight = fp_add(weight, AGGREGATION_LAYER_BIAS_MAT->data[0]);
        
        aggregationWeights.data[VECTOR_COLUMN(i - 1)] = weight;
    }

    // `Free` the stacked state from the data memory
    dataBufferOffset -= stacked.numRows * stacked.numCols;

    // Set the result data to zero.
    matrix_set(result, 0);

    // Normalize the weights using SparseMax
    sparsemax(&aggregationWeights, &aggregationWeights, precision);

    matrix temp = { DATA_BUFFER + dataBufferOffset, NUM_OUTPUT_FEATURES, VECTOR_COLS };
    dataBufferOffset += temp.numRows * temp.numCols;

    // Aggregate logits using the weighted average
    for (i = n; i > 0; i--) {
        // Apply weight
        scalar_product(&temp, logits + i - 1, aggregationWeights.data[VECTOR_COLUMN(i - 1)], precision);

        // Aggregate into the result array
        matrix_add(result, result, &temp);
    }

    return result;
}
#endif


#if defined(IS_BUDGET_RNN) && STRIDE_LENGTH > 1
matrix *fuse_states(matrix *result, matrix *current, matrix *previous, uint16_t precision) {
    /**
     * Combines the given states using a learned weighted average.
     */
    // Allocate intermediate states
    uint16_t data_buffer_offset = 0;

    matrix fusionInput = { DATA_BUFFER + data_buffer_offset, current->numRows + previous->numRows, current->numCols };
    data_buffer_offset += fusionInput.numRows * fusionInput.numCols;

    matrix fusionGate = { DATA_BUFFER + data_buffer_offset, current->numRows, current->numCols };
    data_buffer_offset += fusionGate.numRows * fusionGate.numCols;

    matrix gateTemp = { DATA_BUFFER + data_buffer_offset, current->numRows, current->numCols };
    data_buffer_offset += gateTemp.numRows * gateTemp.numCols;

    // Stack the states
    vstack(&fusionInput, current, previous);

    // Compute the fusion gate
    dense(&fusionGate, &fusionInput, RNN_CELL_W_FUSION_MAT, RNN_CELL_B_FUSION_MAT, &fp_sigmoid, precision);

    apply_gate(result, &fusionGate, current, previous, &gateTemp, precision);
    return result;
}
#endif


#ifdef IS_SKIP_RNN
int16_t binarize_update_prob(int16_t updateProb, uint16_t precision) {
    int16_t half = 1 << (precision - 1);
    int16_t one = 1 << precision;

    if (updateProb >= half) {
        return one;
    }
    return 0;
}


int16_t get_state_update_prob(matrix *state, int16_t prevUpdateProb, uint16_t precision) {
    uint16_t binaryUpdateProb = binarize_update_prob(prevUpdateProb, precision);
    
    int16_t deltaUpdate = dot_product(RNN_CELL_W_STATE_MAT, state, precision);
    int16_t deltaUpdateProb = fp_sigmoid(fp_add(deltaUpdate, RNN_CELL_B_STATE_MAT->data[0]), FIXED_POINT_PRECISION);
    
    int16_t one = 1 << precision;
    if (binaryUpdateProb == one) {
        return deltaUpdateProb;
    }

    int16_t nextUpdateProb = prevUpdateProb + deltaUpdateProb;
    if (nextUpdateProb >= one) {
        return one;
    }
    return nextUpdateProb;
}
#endif

#ifdef IS_PHASED_RNN
int16_t phase_gate(int16_t t, uint16_t precision) {
    // Compute the phi_t function
    int16_t fp_t = int_to_fp(t, precision);

    int16_t period_shift = fp_t - RNN_CELL_SHIFT;
    period_shift = fp_mod(period_shift, RNN_CELL_PERIOD, precision);

    int16_t phi_t = fp_div(period_shift, RNN_CELL_PERIOD, precision);

    int16_t half = 1 << (precision - 1);
    int16_t split = fp_mul(ON_FRACTION, half, precision);

    if (phi_t > ON_FRACTION) {
        return 0;
    }

    int16_t gate_val = fp_mul(int_to_fp(2, precision), phi_t, precision);
    gate_val = fp_div(gate_val, ON_FRACTION, precision);

    if (phi_t < split) {
        return gate_val;
    }

    gate_val = fp_add(int_to_fp(2, precision), fp_neg(gate_val));
    return gate_val;
}
#endif
