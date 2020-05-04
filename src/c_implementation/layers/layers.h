#include <stdint.h>
#include "../math/matrix_ops.h"
#include "../math/matrix.h"
#include "../math/fixed_point_ops.h"
#include "../memory.h"
#include "cells.h"

#define LAYERS_GUARD
#ifndef LAYERS_GUARD

matrix *dense(matrix *result, matrix *input, matrix *W, matrix *b, int16_t (*activation)(int16_t, int16_t), int16_t precision);
matrix *apply_gru(matrix *result, matrix *input, matrix *state, GRU *gru, int16_t precision);
matrix *apply_gate(matrix *result, matrix *gate, matrix *first, matrix *second, int16_t precision);
matrix *rnn(matrix *result, matrix **inputs, void *cell, enum CellType cellType, int16_t seqLength, int16_t precision);

#endif
