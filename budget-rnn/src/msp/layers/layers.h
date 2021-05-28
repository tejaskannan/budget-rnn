#include <stdint.h>
#include "../math/matrix_ops.h"
#include "../math/matrix.h"
#include "../math/fixed_point_ops.h"
#include "../utils/utils.h"
#include "../neural_network_parameters.h"
#include "cells.h"

#ifndef LAYERS_GUARD
#define LAYERS_GUARD

// Standard Neural Network Functions
matrix *dense(matrix *result, matrix *input, matrix *W, matrix *b, int16_t (*activation)(int16_t, uint16_t), uint16_t precision);
matrix *apply_gate(matrix *result, matrix *gate, matrix *first, matrix *second, matrix *temp, uint16_t precision);

// RNN Cell Functions
matrix *apply_gru(matrix *result, matrix *input, matrix *state, GRU *gru, GRUTempStates *temp, uint16_t precision);
matrix *apply_ugrnn(matrix *result, matrix *input, matrix *state, UGRNN *ugrnn, UGRNNTempStates *temp, uint16_t precision);

#endif
