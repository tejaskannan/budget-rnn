#include "neural_network.h"
int8_t *execute_model(matrix *inputs[SEQ_LENGTH], int8_t *outputs) {
	GRU rnn_cell = { TRANSFORM_RNN_CELL_W_UPDATE_KERNEL_0_MAT, TRANSFORM_RNN_CELL_U_UPDATE_KERNEL_0_MAT, TRANSFORM_RNN_CELL_B_UPDATE_BIAS_0_MAT, TRANSFORM_RNN_CELL_W_RESET_KERNEL_0_MAT, TRANSFORM_RNN_CELL_U_RESET_KERNEL_0_MAT, TRANSFORM_RNN_CELL_B_RESET_BIAS_0_MAT, TRANSFORM_RNN_CELL_W_KERNEL_0_MAT, TRANSFORM_RNN_CELL_U_KERNEL_0_MAT, TRANSFORM_RNN_CELL_B_BIAS_0_MAT };


	matrix transformedMat;
	dtype transformedData[20];
	transformedMat.data = transformedData;
	transformedMat.numRows = 20;
	transformedMat.numCols = 1;
	matrix *transformed = &transformedMat;

	matrix stateMat;
	dtype stateData[20];
	stateMat.data = stateData;
	stateMat.numRows = 20;
	stateMat.numCols = 1;
	matrix *state = &stateMat;

	matrix temp_stateMat;
	dtype temp_stateData[20];
	temp_stateMat.data = temp_stateData;
	temp_stateMat.numRows = 20;
	temp_stateMat.numCols = 1;
	matrix *temp_state = &temp_stateMat;

	matrix fusion_gateMat;
	dtype fusion_gateData[20];
	fusion_gateMat.data = fusion_gateData;
	fusion_gateMat.numRows = 20;
	fusion_gateMat.numCols = 1;
	matrix *fusion_gate = &fusion_gateMat;

	matrix gateTempMat;
	dtype gateTempData[20];
	gateTempMat.data = gateTempData;
	gateTempMat.numRows = 20;
	gateTempMat.numCols = 1;
	matrix *gateTemp = &gateTempMat;

	matrix fusion_stackMat;
	dtype fusion_stackData[40];
	fusion_stackMat.data = fusion_stackData;
	fusion_stackMat.numRows = 40;
	fusion_stackMat.numCols = 1;
	matrix *fusion_stack = &fusion_stackMat;

	matrix outputTemp0Mat;
	dtype outputTemp0Data[32];
	outputTemp0Mat.data = outputTemp0Data;
	outputTemp0Mat.numRows = 32;
	outputTemp0Mat.numCols = 1;
	matrix *outputTemp0 = &outputTemp0Mat;

	matrix outputTemp1Mat;
	dtype outputTemp1Data[32];
	outputTemp1Mat.data = outputTemp1Data;
	outputTemp1Mat.numRows = 32;
	outputTemp1Mat.numCols = 1;
	matrix *outputTemp1 = &outputTemp1Mat;

	matrix outputMat;
	dtype outputData[6];
	outputMat.data = outputData;
	outputMat.numRows = 6;
	outputMat.numCols = 1;
	matrix *output = &outputMat;

	matrix *prev_states[SAMPLES_PER_SEQ];
	matrix prevStates0Mat;
	dtype prevStates0Data[20];
	prevStates0Mat.data = prevStates0Data;
	prevStates0Mat.numRows = 20;
	prevStates0Mat.numCols = 1;
	matrix *prevStates0 = &prevStates0Mat;
	prev_states[0] = prevStates0;

	matrix prevStates1Mat;
	dtype prevStates1Data[20];
	prevStates1Mat.data = prevStates1Data;
	prevStates1Mat.numRows = 20;
	prevStates1Mat.numCols = 1;
	matrix *prevStates1 = &prevStates1Mat;
	prev_states[1] = prevStates1;

	GRUTempStates gruTemp;
	matrix gruTempUpdateMat;
	dtype gruTempUpdateData[20];
	gruTempUpdateMat.data = gruTempUpdateData;
	gruTempUpdateMat.numRows = 20;
	gruTempUpdateMat.numCols = 1;
	matrix *gruTempUpdate = &gruTempUpdateMat;
	gruTemp.update = gruTempUpdate;

	matrix gruTempResetMat;
	dtype gruTempResetData[20];
	gruTempResetMat.data = gruTempResetData;
	gruTempResetMat.numRows = 20;
	gruTempResetMat.numCols = 1;
	matrix *gruTempReset = &gruTempResetMat;
	gruTemp.reset = gruTempReset;

	matrix gruTempCandidateMat;
	dtype gruTempCandidateData[20];
	gruTempCandidateMat.data = gruTempCandidateData;
	gruTempCandidateMat.numRows = 20;
	gruTempCandidateMat.numCols = 1;
	matrix *gruTempCandidate = &gruTempCandidateMat;
	gruTemp.candidate = gruTempCandidate;

	matrix gruTempInputtempMat;
	dtype gruTempInputtempData[20];
	gruTempInputtempMat.data = gruTempInputtempData;
	gruTempInputtempMat.numRows = 20;
	gruTempInputtempMat.numCols = 1;
	matrix *gruTempInputtemp = &gruTempInputtempMat;
	gruTemp.inputTemp = gruTempInputtemp;

	matrix gruTempGatetempMat;
	dtype gruTempGatetempData[20];
	gruTempGatetempMat.data = gruTempGatetempData;
	gruTempGatetempMat.numRows = 20;
	gruTempGatetempMat.numCols = 1;
	matrix *gruTempGatetemp = &gruTempGatetempMat;
	gruTemp.gateTemp = gruTempGatetemp;

	int16_t i, j;
	for (i = 0; i < NUM_SEQUENCES; i++) {
		matrix_set(state, 0);
		for (j = 0; j < SAMPLES_PER_SEQ; j++) {
			matrix *input = inputs[j * 10 + i];
			transformed = dense(transformed, input, EMBEDDING_KERNEL_0_MAT, EMBEDDING_BIAS_0_MAT, &fp_relu, IS_COMPRESSED, "em", FIXED_POINT_PRECISION);
			if (i > 0) {
				fusion_stack = stack(fusion_stack, state, prev_states[j]);
				fusion_gate = dense(fusion_gate, fusion_stack, FUSION_KERNEL_0_MAT, FUSION_BIAS_0_MAT, &fp_linear, IS_COMPRESSED, "fs0", FIXED_POINT_PRECISION);
				temp_state = apply_gate(temp_state, fusion_gate, state, prev_states[j], gateTemp, FIXED_POINT_PRECISION);
				state = matrix_replace(state, temp_state);
			}
			temp_state = apply_gru(temp_state, transformed, state, &rnn_cell, &gruTemp, IS_COMPRESSED, 0, FIXED_POINT_PRECISION);
			state = matrix_replace(state, temp_state);
			matrix_replace(prev_states[j], state);
		}

		outputTemp0 = dense(outputTemp0, state, OUTPUT_HIDDEN_KERNEL_0_MAT, OUTPUT_HIDDEN_BIAS_0_MAT, &fp_relu, IS_COMPRESSED, "ou0", FIXED_POINT_PRECISION);
		outputTemp1 = dense(outputTemp1, outputTemp0, OUTPUT_HIDDEN_KERNEL_1_MAT, OUTPUT_HIDDEN_BIAS_1_MAT, &fp_relu, IS_COMPRESSED, "ou1", FIXED_POINT_PRECISION);
		output = dense(output, outputTemp1, OUTPUT_KERNEL_0_MAT, OUTPUT_BIAS_0_MAT, &fp_linear, IS_COMPRESSED, "ou", FIXED_POINT_PRECISION);

		int16_t prediction = argmax(output);
		outputs[i] = prediction;
	}

	return outputs;
}
