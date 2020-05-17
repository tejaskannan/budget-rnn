#include <stdint.h>

#ifndef MEMORY_GUARD
#define MEMORY_GUARD

// Constants for memory allocation. We represent memory as an array of bytes.
#define MEMORY_BYTES 3500

typedef uint16_t Header;
static const uint16_t HEADER_SIZE = sizeof(Header);

static int8_t MEMORY[MEMORY_BYTES];
static void *NULL_PTR = (void *) 0x0;  // Custom-defined null pointer.

int8_t *alloc(uint16_t numBytes);
void dealloc(void *ptr);
int8_t isNull(void *ptr);
uint16_t freeBytes(void);
uint16_t allocBytes(void);
void clearMemory(void);

// Functions to handle headers
uint8_t isFree(Header header);
Header createHeader(uint16_t numBytes, uint8_t allocBit);
uint8_t getAllocBit(Header header);
Header setAllocBit(Header header, uint8_t allocBit);
uint16_t getAllocSize(Header header);
Header setAllocSize(Header header, uint16_t allocSize);


#endif
