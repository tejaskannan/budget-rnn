#include "neural_network.h"
int16_t *execute_model(matrix *inputs[SEQ_LENGTH], int16_t *outputs) {
	GRU rnn_cell = { TRANSFORM_RNN_CELL_W_UPDATE_0_MAT, TRANSFORM_RNN_CELL_U_UPDATE_0_MAT, TRANSFORM_RNN_CELL_B_UPDATE_0_MAT, TRANSFORM_RNN_CELL_W_RESET_0_MAT, TRANSFORM_RNN_CELL_U_RESET_0_MAT, TRANSFORM_RNN_CELL_B_RESET_0_MAT, TRANSFORM_RNN_CELL_W_0_MAT, TRANSFORM_RNN_CELL_U_0_MAT, TRANSFORM_RNN_CELL_B_0_MAT };

	matrix *transformed = matrix_allocate(16, 1);
	matrix *state = matrix_allocate(16, 1);
	matrix *temp_state = matrix_allocate(16, 1);
	matrix *fusion_stack = matrix_allocate(2 * 16, 1);
	matrix *fusion_gate = matrix_allocate(16, 1);

	matrix *output = matrix_allocate(OUTPUT_KERNEL_0_MAT->numRows, 1);

	matrix *prev_states[SAMPLES_PER_SEQ];
	for (int16_t i = 0; i < SAMPLES_PER_SEQ; i++) {
		prev_states[i] = matrix_allocate(16, 1);
	}

	for (int16_t i = 0; i < 5; i++) {
		matrix_set(state, 0);
		for (int16_t j = 0; j < SAMPLES_PER_SEQ; j++) {
			matrix *input = inputs[j * 5 + i];
			transformed = dense(transformed, input, EMBEDDING_KERNEL_0_MAT, EMBEDDING_BIAS_0_MAT, &fp_tanh, FIXED_POINT_PRECISION);
			if (i > 0) {
				fusion_stack = stack(fusion_stack, state, prev_states[j]);
				fusion_gate = dense(fusion_gate, fusion_stack, FUSION_KERNEL_0_MAT, FUSION_BIAS_0_MAT, &fp_linear, FIXED_POINT_PRECISION);
				temp_state = apply_gate(temp_state, fusion_gate, state, prev_states[j], FIXED_POINT_PRECISION);
				state = matrix_replace(state, temp_state);
			}
			temp_state = apply_gru(temp_state, transformed, state, &rnn_cell, FIXED_POINT_PRECISION);
			state = matrix_replace(state, temp_state);
			matrix_replace(prev_states[j], state);
		}

		matrix *temp0 = matrix_allocate(OUTPUT_HIDDEN_KERNEL_0_MAT->numRows, state->numCols);
		temp0 = dense(temp0, state, OUTPUT_HIDDEN_KERNEL_0_MAT, OUTPUT_HIDDEN_BIAS_0_MAT, &fp_tanh, FIXED_POINT_PRECISION);
		matrix *temp1 = matrix_allocate(OUTPUT_HIDDEN_KERNEL_1_MAT->numRows, temp0->numCols);
		temp1 = dense(temp1, temp0, OUTPUT_HIDDEN_KERNEL_1_MAT, OUTPUT_HIDDEN_BIAS_1_MAT, &fp_tanh, FIXED_POINT_PRECISION);
		output = dense(output, temp1, OUTPUT_KERNEL_0_MAT, NULL_PTR, &fp_linear, FIXED_POINT_PRECISION);
		matrix_free(temp0);
		matrix_free(temp1);

		int16_t prediction = argmax(output);
		outputs[i] = prediction;
	}

	matrix_free(transformed);
	matrix_free(state);
	matrix_free(fusion_gate);
	matrix_free(fusion_stack);
	matrix_free(output);
	matrix_free(temp_state);
	for (int16_t i = 0; i < SAMPLES_PER_SEQ; i++) {
		matrix_free(prev_states[i]);
	}

	return outputs;
}
