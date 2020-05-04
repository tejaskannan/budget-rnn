#include "../math/matrix.h"

#ifndef CELLS_GUARD
#define CELLS_GUARD

struct GRU {
    matrix *wUpdate;
    matrix *uUpdate;
    matrix *bUpdate;
    matrix *wReset;
    matrix *uReset;
    matrix *bReset;
    matrix *wCandidate;
    matrix *uCandidate;
    matrix *bCandidate;
};
typedef struct GRU GRU;


struct TFGRU {
    matrix *wGates;
    matrix *bGates;
    matrix *wCandidates;
    matrix *bCandidates;
};
typedef struct TFGRU TFGRU;


// Supported Cell Types
enum CellType { GRUCell = 0, TFGRUCell = 1 };

#endif
