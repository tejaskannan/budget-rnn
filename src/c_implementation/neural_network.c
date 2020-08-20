#include "neural_network.h"

// Buffer for intermediate states
dtype DATA_BUFFER[400];


uint16_t getCurrentLevel(uint16_t seqIndex) {
    #ifndef IS_ADAPTIVE
    return seqIndex;
    #endif
    
    uint16_t numSequences = SEQ_LENGTH / SAMPLES_PER_SEQ;

    #ifdef IS_SAMPLE
    return seqIndex % numSequences;
    #else
    return seqIndex / SAMPLES_PER_SEQ;
    #endif
}


uint8_t isLastSampleInLevel(uint16_t seqIndex, uint16_t fixedSeqIndex) {
    #ifndef IS_ADAPTIVE
        return seqIndex == fixedSeqIndex;
    #elif defined(IS_SAMPLE)
        UNUSED(fixedSeqIndex);
        return (SEQ_LENGTH - seqIndex) <= (SEQ_LENGTH / SAMPLES_PER_SEQ);
    #else
        UNUSED(fixedSeqIndex);
        return ((seqIndex + 1) % SAMPLES_PER_SEQ) == 0;
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
    dense(&embedding, input, EMBEDDING_KERNEL_0_MAT, EMBEDDING_BIAS_0_MAT, &fp_leaky_relu, precision);

    // Apply the specified transformation layer
    #ifdef GRU_TRANSFORM
        // Allocate temporary states
        matrix inputTemp = { DATA_BUFFER + data_buffer_offset, state->numRows, state->numCols };
        data_buffer_offset += inputTemp.numRows * inputTemp.numCols;

        matrix update = { DATA_BUFFER + data_buffer_offset, state->numRows, state->numCols };
        data_buffer_offset += update.numRows * update.numCols;

        matrix reset = { DATA_BUFFER + data_buffer_offset, state->numRows, state->numCols };
        data_buffer_offset += reset.numRows * reset.numCols;

        matrix candidate = { DATA_BUFFER + data_buffer_offset, state->numRows, state->numCols };
        data_buffer_offset += candidate.numRows * candidate.numCols;

        matrix gateTemp = { DATA_BUFFER + data_buffer_offset, state->numRows, state->numCols };
        data_buffer_offset += gateTemp.numRows * gateTemp.numCols;

        GRUTempStates rnnTemp;
        rnnTemp.inputTemp = &inputTemp;
        rnnTemp.update = &update;
        rnnTemp.reset = &reset;
        rnnTemp.candidate = &candidate;
        rnnTemp.gateTemp = &gateTemp;

        /// Create the GRU Cell
	    GRU rnn_cell = { TRANSFORM_RNN_CELL_W_UPDATE_KERNEL_0_MAT, TRANSFORM_RNN_CELL_U_UPDATE_KERNEL_0_MAT, TRANSFORM_RNN_CELL_B_UPDATE_BIAS_0_MAT, TRANSFORM_RNN_CELL_W_RESET_KERNEL_0_MAT, TRANSFORM_RNN_CELL_U_RESET_KERNEL_0_MAT, TRANSFORM_RNN_CELL_B_RESET_BIAS_0_MAT, TRANSFORM_RNN_CELL_W_KERNEL_0_MAT, TRANSFORM_RNN_CELL_U_KERNEL_0_MAT, TRANSFORM_RNN_CELL_B_BIAS_0_MAT };
	
        // Apply the GRU Cell
        apply_gru(result, &embedding, state, &rnn_cell, &rnnTemp, precision);
    #elif defined(UGRNN_TRANSFORM)
        // Allocate temporary states
        matrix inputTemp = { DATA_BUFFER + data_buffer_offset, state->numRows, state->numCols };
        data_buffer_offset += inputTemp.numRows * inputTemp.numCols;

        matrix update = { DATA_BUFFER + data_buffer_offset, state->numRows, state->numCols };
        data_buffer_offset += update.numRows * update.numCols;

        matrix candidate = { DATA_BUFFER + data_buffer_offset, state->numRows, state->numCols };
        data_buffer_offset += candidate.numRows * candidate.numCols;

        matrix gateTemp = { DATA_BUFFER + data_buffer_offset, state->numRows, state->numCols };
        data_buffer_offset += gateTemp.numRows * gateTemp.numCols;

        UGRNNTempStates rnnTemp;
        rnnTemp.inputTemp = &inputTemp;
        rnnTemp.update = &update;
        rnnTemp.candidate = &candidate;
        rnnTemp.gateTemp = &gateTemp;

        /// Create the UGRNN Cell
	    UGRNN rnn_cell = { TRANSFORM_RNN_CELL_W_UPDATE_KERNEL_0_MAT, TRANSFORM_RNN_CELL_U_UPDATE_KERNEL_0_MAT, TRANSFORM_RNN_CELL_B_UPDATE_BIAS_0_MAT, TRANSFORM_RNN_CELL_W_KERNEL_0_MAT, TRANSFORM_RNN_CELL_U_KERNEL_0_MAT, TRANSFORM_RNN_CELL_B_BIAS_0_MAT };

        // Apply the GRU Cell
        apply_ugrnn(result, &embedding, state, &rnn_cell, &rnnTemp, precision);
    #else  // Dense Transformation
        // By default, we use a dense layer with two hidden layers
        // Allocate intermediate states
        matrix hidden0 = { DATA_BUFFER + data_buffer_offset, TRANSFORM_HIDDEN_BIAS_0_MAT->numRows, TRANSFORM_HIDDEN_BIAS_0_MAT->numCols };
        data_buffer_offset += hidden0.numRows * hidden0.numCols;

        matrix hidden1 = { DATA_BUFFER + data_buffer_offset, TRANSFORM_HIDDEN_BIAS_1_MAT->numRows, TRANSFORM_HIDDEN_BIAS_1_MAT->numCols };
        data_buffer_offset += hidden1.numRows * hidden1.numCols;

        // Apply hidden layers
        dense(&hidden0, &embedding, TRANSFORM_HIDDEN_KERNEL_0_MAT, TRANSFORM_HIDDEN_BIAS_0_MAT, &fp_leaky_relu, precision);
        dense(&hidden1, &hidden0, TRANSFORM_HIDDEN_KERNEL_1_MAT, TRANSFORM_HIDDEN_BIAS_1_MAT, &fp_leaky_relu, precision);
        
        // Apply the output layer
        dense(result, &hidden1, TRANSFORM_KERNEL_0_MAT, TRANSFORM_BIAS_0_MAT, &fp_leaky_relu, precision);
    #endif

    return result;
}


// Function to compute output (1 or 2 hidden layer depending on model type)
int16_t compute_prediction(matrix *input, uint16_t precision) {
    /**
     * Function to compute the prediction using a feed-forward network.
     */
    // Allocate intermediate states
    uint16_t data_buffer_offset = 0;
    matrix hidden = { DATA_BUFFER + data_buffer_offset, OUTPUT_HIDDEN_BIAS_0_MAT->numRows, 1 };
    data_buffer_offset += hidden.numRows * hidden.numCols;

    matrix output = { DATA_BUFFER + data_buffer_offset, NUM_OUTPUT_FEATURES, 1 };
    data_buffer_offset += output.numRows * output.numCols;

    // Apply the dense layers
    dense(&hidden, input, OUTPUT_HIDDEN_KERNEL_0_MAT, OUTPUT_HIDDEN_BIAS_0_MAT, &fp_leaky_relu, precision);

    #ifdef IS_RNN
    dense(&output, &hidden, OUTPUT_KERNEL_0_MAT, OUTPUT_BIAS_0_MAT, &fp_linear, precision);
    #else
    matrix hidden1 = { DATA_BUFFER + data_buffer_offset, OUTPUT_HIDDEN_BIAS_1_MAT->numRows, 1 };
    dense(&hidden1, &hidden, OUTPUT_HIDDEN_KERNEL_1_MAT, OUTPUT_HIDDEN_BIAS_1_MAT, &fp_leaky_relu, precision);
    dense(&output, &hidden1, OUTPUT_KERNEL_0_MAT, NULL_PTR, &fp_linear, precision);
    #endif

    return argmax(&output);
}


#ifdef IS_ADAPTIVE
int16_t compute_stop_output(matrix *state, uint16_t precision) {
    /**
     * Computes the stop output from the given state.
     */
    // Allocate intermediate states
    uint16_t data_buffer_offset = 0;
    matrix hidden = { DATA_BUFFER + data_buffer_offset, STOP_PREDICTION_HIDDEN_BIAS_0_MAT->numRows, 1 };
    data_buffer_offset += hidden.numRows * hidden.numCols;

    matrix output = { DATA_BUFFER + data_buffer_offset, 1, 1 };
    data_buffer_offset += output.numRows * output.numCols;

    // Apply the dense layers
    dense(&hidden, state, STOP_PREDICTION_HIDDEN_KERNEL_0_MAT, STOP_PREDICTION_HIDDEN_BIAS_0_MAT, &fp_leaky_relu, precision);
    dense(&output, &hidden, STOP_PREDICTION_KERNEL_0_MAT, STOP_PREDICTION_BIAS_0_MAT, &fp_sigmoid, precision);

    return output.data[0];
}
#endif


#if defined(IS_SAMPLE) && defined(IS_RNN)
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
    stack(&fusionInput, current, previous);

    // Compute the fusion gate
    dense(&fusionGate, &fusionInput, FUSION_KERNEL_0_MAT, FUSION_BIAS_0_MAT, &fp_sigmoid, precision);

    apply_gate(result, &fusionGate, current, previous, &gateTemp, precision);
    return result;
}
#endif


// Function to pool states in NBOW models
#ifndef IS_RNN
void normalizeSampling(dtype normalizedWeights[SEQ_LENGTH], dtype weights[SEQ_LENGTH], uint16_t n, uint16_t precision) {
    uint16_t sampleLevel = 0;

    // Normalize the weights
    dtype weight_sum = 0;
    uint16_t i = 0;
    for (; i < SEQ_LENGTH; i++) {
        sampleLevel = getCurrentLevel(i);
        if (sampleLevel <= n) {
            weight_sum += weights[i];
        }
    }
    
    for (i = 0; i < SEQ_LENGTH; i++) {
        sampleLevel = getCurrentLevel(i);
        if (sampleLevel <= n) {
            normalizedWeights[i] = fp_div(weights[i], weight_sum, precision);
        } else {
            normalizedWeights[i] = 0;
        }
    }
}

void normalizeConsecutive(dtype normalizedWeights[SEQ_LENGTH], dtype weights[SEQ_LENGTH], uint16_t n, uint16_t precision) {
    uint16_t sampleLevel = 0;

    // Normalize the weights
    dtype weight_sum = 0;
    uint16_t i = 0;
    for (; i <= n; i++) {
        weight_sum += weights[i];
    }
    
    for (i = 0; i < SEQ_LENGTH; i++) {
        if (i <= n) {
            normalizedWeights[i] = fp_div(weights[i], weight_sum, precision);
        } else {
            normalizedWeights[i] = 0;
        }
    }
}

matrix *pool_states(matrix *result, matrix states[SEQ_LENGTH], dtype weights[SEQ_LENGTH], uint16_t n, uint8_t useSampleStrategy, uint16_t precision){
    uint8_t mask = 1;
    uint16_t sampleLevel = 0;

    // Normalize the weights
    dtype normalizedWeights[SEQ_LENGTH];
    if (useSampleStrategy) {
        normalizeSampling(normalizedWeights, weights, n, precision);
    } else {
        normalizeConsecutive(normalizedWeights, weights, n, precision);
    }

    // Zero out the result before aggregation
    matrix_set(result, 0);

    // Create the temp state
    matrix temp = { DATA_BUFFER, result->numRows, result->numCols };

    uint16_t i;
    for (i = 0; i < SEQ_LENGTH; i++) {
        // Apply the normalized weight
        scalar_product(&temp, states + i, normalizedWeights[i], precision);

        // Accumulate into the result matrix
        matrix_add(result, result, &temp);
    }

    return result;
}


dtype compute_aggregation_weight(matrix *state, uint16_t precision) {
    // Allocate intermediate state
    uint16_t data_buffer_offset = 0;

    matrix weight = { DATA_BUFFER + data_buffer_offset, 1, 1 };
    data_buffer_offset += weight.numRows * weight.numCols;

    dense(&weight, state, AGGREGATE_KERNEL_0_MAT, AGGREGATE_BIAS_0_MAT, &fp_sigmoid, precision);

    return weight.data[0];
}
#endif
