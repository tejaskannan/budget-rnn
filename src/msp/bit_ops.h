#ifndef BIT_OPS_GUARD
#define BIT_OPS_GUARD

#define SET_BIT(X, Y)   ((X) |= (Y))
#define CLR_BIT(X, Y)   ((X) &= ~(Y))
#define TGL_BIT(X, Y)   ((X) ^= (Y))
#define TEST_BIT(X, Y)  ((X) & (Y))

#endif
