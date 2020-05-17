#include <stdint.h>

#ifndef MATRIX_GUARD
    #define MATRIX_GUARD

    typedef int16_t dtype;

    struct matrix {
        int8_t numRows;
        int8_t numCols;
        dtype *data;
    };
    typedef struct matrix matrix;

#endif
