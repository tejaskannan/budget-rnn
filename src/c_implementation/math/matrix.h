#include <stdint.h>

#ifndef MATRIX_GUARD
    #define MATRIX_GUARD

    struct matrix {
        int8_t numRows;
        int8_t numCols;
        int16_t *data;
    };
    typedef struct matrix matrix;

#endif
