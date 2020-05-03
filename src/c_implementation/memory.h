#include <stdint.h>

#ifndef MEMORY_GUARD
#define MEMORY_GUARD

// Constants for memory allocation
#define MEMORY_BYTES 3500
#define HEADER_SIZE 1
static int8_t MEMORY[MEMORY_BYTES];
static void *NULL_PTR = (void *) 0x0;

int8_t *alloc(int8_t numBytes);
void free(void *ptr);
int8_t isNull(void *ptr);
uint16_t freeBytes(void);
uint16_t allocBytes(void);

#endif
