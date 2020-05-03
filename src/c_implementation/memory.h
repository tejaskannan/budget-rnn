#include <stdint.h>

#ifndef MEMORY_GUARD
#define MEMORY_GUARD

// Constants for memory allocation
#define MEMORY_BYTES  3500
#define HEADER_SIZE 1
static int8_t MEMORY[MEMORY_BYTES];
static int8_t *NULL_PTR = (int8_t *) 0x0;

int8_t *alloc(int8_t numBytes);
void free(void *ptr);
int8_t isNull(void *ptr);
int16_t freeBytes(void);
int16_t allocBytes(void);

#endif


