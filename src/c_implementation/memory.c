#include "memory.h"


int8_t isNull(void *ptr) {
    return ((int8_t *) ptr) == NULL_PTR;
}


int8_t *alloc(uint8_t numBytes) {
    if (numBytes == 0) {
        return NULL_PTR;
    }

    // Use the header elements to track the length of allocated blocks.
    uint16_t i = 1;
    while (i < MEMORY_BYTES) {
        if (!MEMORY[i]) {
            int8_t isValidSpace = 1;
            uint16_t j = i;
            for (; j < i + numBytes + HEADER_SIZE; j++) {
                // The first non-zero 
                if (j >= MEMORY_BYTES || MEMORY[j])  {
                    isValidSpace = 0;
                    break;
                }
            }

            // If the space has a valid number of bytes, then we can allocate the new block here.
            // Otherwise, continue from this point on.
            if (isValidSpace) {
                break;
            } else {
                i = j;
            }
        } else {
            // This is a block header, so we skip across
            // the size of the allocation
            i += ((uint8_t) MEMORY[i]) + HEADER_SIZE;
        }
    }

    if (i >= MEMORY_BYTES) {
        return NULL_PTR;
    }

    // We have reached a free block. We allocate it using the header - data format.
    MEMORY[i] = numBytes;
    return MEMORY + i + HEADER_SIZE;  // The data block starts one byte after the header.
}


void free(void *ptr) {
    // The pointer points to the beginning of the data block. The header is located exactly one
    // byte behind this. We use this to mark the block as freed by setting the number of allocated.
    // bytes to zero.
    int8_t *bytePtr = (int8_t *) ptr;
    uint8_t numBytes = (uint8_t) *(bytePtr - HEADER_SIZE);
    *(bytePtr - HEADER_SIZE) = 0;
    
    // For convenience, we zero out the freed memory.
    int8_t i;
    for (i = 0; i < numBytes; i++) {
        bytePtr[i] = 0;
    }
}


uint16_t allocBytes(void) {
    uint16_t bytes = 0;
    
    // Traverse the block headers to determine the number of allocated bytes
    int16_t i = 0;
    while (i < MEMORY_BYTES) {
        uint8_t headerValue = (uint8_t) MEMORY[i];

        if (headerValue) {
            bytes += headerValue + HEADER_SIZE;  // Account for the header
            i += headerValue + HEADER_SIZE;
        } else {
            i += 1;
        }
    }

    return bytes;
}


uint16_t freeBytes(void) {
    return MEMORY_BYTES - allocBytes();
}
