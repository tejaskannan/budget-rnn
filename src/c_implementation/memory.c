#include "memory.h"


int8_t isNull(void *ptr) {
    return ((int8_t *) ptr) == NULL_PTR;
}


int8_t *alloc(int8_t numBytes) {
    if (numBytes <= 0) {
        return NULL_PTR;
    }

    // Use the header elements to track the length of allocated blocks.
    int16_t i = 0;
    while (MEMORY[i]) {
        i += MEMORY[i] + HEADER_SIZE;  // Add one to account for the header
        if (i >= MEMORY_BYTES) {
            return NULL_PTR;
        }
    }

    // We have reached a free block. We allocate it using the header - data structure.
    MEMORY[i] = numBytes;
    return MEMORY + i + HEADER_SIZE;  // The data block starts one byte after the header.
}


void free(void *ptr) {
    // The pointer points to the beginning of the data block. The header is located exactly one
    // byte behind this. We use this to mark the block as freed by setting the number of allocated.
    // bytes to zero.
    int8_t *bytePtr = (int8_t *) ptr;
    int8_t numBytes = *(bytePtr - HEADER_SIZE);
    *(bytePtr - HEADER_SIZE) = 0;
    
    // For convenience, we zero out the freed memory.
    int8_t i;
    for (i = 0; i < numBytes; i++) {
        bytePtr[i] = 0;
    }
}


int16_t allocBytes(void) {
    int16_t bytes = 0;
    
    // Traverse the block headers to determine the number of allocated bytes
    int16_t i = 0;
    while (i < MEMORY_BYTES) {
        if (MEMORY[i]) {
            bytes += MEMORY[i] + HEADER_SIZE;  // Account for the header
            i += MEMORY[i] + HEADER_SIZE;
        } else {
            i += 1;
        }
    }

    return bytes;
}


int16_t freeBytes(void) {
    return MEMORY_BYTES - allocBytes();
}






