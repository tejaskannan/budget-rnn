#include <stdint.h>
#include "../math/matrix_ops.h"
#include "../math/matrix.h"
#include "../math/fixed_point_ops.h"
#include "../utils/seeds.h"
#include "../memory.h"
#include "cells.h"

#ifndef LAYERS_GUARD
#define LAYERS_GUARD

matrix *dense(matrix *result, matrix *input, matrix *W, matrix *b, int16_t (*activation)(int16_t, uint16_t), uint16_t precision);

matrix *apply_gru(matrix *result, matrix *input, matrix *state, GRU *gru, GRUTempStates *temp, uint16_t precision);
matrix *apply_ugrnn(matrix *result, matrix *input, matrix *state, UGRNN *ugrnn, UGRNNTempStates *temp, uint16_t precision);
matrix *apply_tf_gru(matrix *result, matrix *input, matrix *state, TFGRU *gru, TFGRUTempStates *tempStates, uint16_t precision);

matrix *apply_gate(matrix *result, matrix *gate, matrix *first, matrix *second, matrix *temp, uint16_t precision);

#endif
