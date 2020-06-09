#include <stdint.h>
#include "../math/matrix_ops.h"
#include "../math/matrix.h"
#include "../math/fixed_point_ops.h"
#include "../utils/seeds.h"
#include "../memory.h"
#include "cells.h"

#ifndef LAYERS_GUARD
#define LAYERS_GUARD

matrix *dense(matrix *result, matrix *input, matrix *W, matrix *b, int16_t (*activation)(int16_t, int16_t), uint8_t is_compressed, char *seed, int16_t precision);

matrix *apply_gru(matrix *result, matrix *input, matrix *state, GRU *gru, GRUTempStates *temp, uint8_t is_compressed, uint8_t layer, int16_t precision);
matrix *apply_tf_gru(matrix *result, matrix *input, matrix *state, TFGRU *gru, int16_t precision);

matrix *apply_gate(matrix *result, matrix *gate, matrix *first, matrix *second, matrix *temp, int16_t precision);
matrix *rnn(matrix *result, matrix **inputs, void *cell, enum CellType cellType, int16_t seqLength, int16_t precision);

#endif
