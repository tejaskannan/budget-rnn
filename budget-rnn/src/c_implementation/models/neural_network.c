#include "neural_network.h"
int16_t execute_model(matrix *inputs[SEQ_LENGTH]) {
	TFGRU rnn_cell = { TRANSFORM_LAYER_CELL_GATES_KERNEL_0_0_MAT, TRANSFORM_LAYER_CELL_GATES_BIAS_0_0_MAT, TRANSFORM_LAYER_CELL_CANDIDATE_KERNEL_0_0_MAT, TRANSFORM_LAYER_CELL_CANDIDATE_BIAS_0_0_MAT };

	matrix *transformed = matrix_allocate(16, 1);
	matrix *state = matrix_allocate(16, 1);
	matrix_set(state, 0);
	matrix *temp_state = matrix_allocate(16, 1);

	for (int16_t i = 0; i < SEQ_LENGTH; i++) {
		matrix *input = inputs[i];
		transformed = dense(transformed, input, EMBEDDING_KERNEL_0_MAT, EMBEDDING_BIAS_0_MAT, &fp_tanh, FIXED_POINT_PRECISION);
		temp_state = apply_tf_gru(temp_state, transformed, state, &rnn_cell, FIXED_POINT_PRECISION);
		state = matrix_replace(state, temp_state);
	}

	matrix *output = matrix_allocate(OUTPUT_KERNEL_0_MAT->numRows, 1);
	matrix *temp0 = matrix_allocate(OUTPUT_HIDDEN_KERNEL_0_MAT->numRows, state->numCols);
	temp0 = dense(temp0, state, OUTPUT_HIDDEN_KERNEL_0_MAT, OUTPUT_HIDDEN_BIAS_0_MAT, &fp_tanh, FIXED_POINT_PRECISION);
	matrix *temp1 = matrix_allocate(OUTPUT_HIDDEN_KERNEL_1_MAT->numRows, temp0->numCols);
	temp1 = dense(temp1, temp0, OUTPUT_HIDDEN_KERNEL_1_MAT, OUTPUT_HIDDEN_BIAS_1_MAT, &fp_tanh, FIXED_POINT_PRECISION);
	output = dense(output, temp1, OUTPUT_KERNEL_0_MAT, NULL_PTR, &fp_linear, FIXED_POINT_PRECISION);
	matrix_free(temp0);
	matrix_free(temp1);

	int16_t prediction = argmax(output);
	matrix_free(transformed);
	matrix_free(state);
	matrix_free(temp_state);
	matrix_free(output);

	return prediction;
}