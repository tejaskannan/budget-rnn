#include <stdint.h>

#ifndef MATRIX_GUARD
    #define MATRIX_GUARD

    typedef int16_t dtype;

    struct matrix {
        dtype *data;
        int8_t numRows;
        int8_t numCols;
    };
    typedef struct matrix matrix;

#endif
