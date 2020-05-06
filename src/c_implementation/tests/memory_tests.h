#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "../memory.h"

#ifndef MEMORY_TEST_GUARD
#define MEMORY_TEST_GUARD

struct Point {
    int16_t x;
    int16_t y;
};
typedef struct Point Point;


void test_alloc_free(void);
void test_store_byte(void);
void test_pair(void);
void test_triple(void);
void test_struct(void);
void test_nested(void);

#endif
