#include "../math/matrix.h"

#ifndef CELLS_GUARD
#define CELLS_GUARD

struct GRU {
    matrix *wGates;
    matrix *bGates;
    matrix *wCandidate;
    matrix *bCandidate;
};
typedef struct GRU GRU;


struct GRUTempStates {
    matrix *stacked;
    matrix *gates;
    matrix *candidate;
    matrix *gateTemp;
};
typedef struct GRUTempStates GRUTempStates;


struct UGRNN {
    matrix *wTransform;
    matrix *bTransform;
};
typedef struct UGRNN UGRNN;


struct UGRNNTempStates {
    matrix *stacked;
    matrix *transformed;
    matrix *gateTemp;
};
typedef struct UGRNNTempStates UGRNNTempStates;

#endif
