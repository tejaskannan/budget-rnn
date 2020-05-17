#include "memory.h"


int8_t isNull(void *ptr) {
    return ((int8_t *) ptr) == NULL_PTR;
}


uint8_t getAllocBit(Header header) {
    uint8_t allocBit = header & 1;
    return allocBit;
}


Header createHeader(uint16_t numBytes, uint8_t allocBit) {
    Header header = 0;
    header = setAllocSize(header, numBytes);
    return setAllocBit(header, allocBit);
}


Header setAllocBit(Header header, uint8_t allocBit) {
    header = header & ~1;  // Clear the bottom bit
    return header | allocBit;  // Set the bottom bit
}


uint8_t isFree(Header header) {
    return (uint8_t) (getAllocBit(header) == 0);
}


uint16_t getAllocSize(Header header) {
    return (uint16_t) (header >> 1);
}

Header setAllocSize(Header header, uint16_t allocSize) {
    uint8_t allocBit = getAllocBit(header);
    return setAllocBit(allocSize << 1, allocBit);
}


int8_t *alloc(uint16_t numBytes) {
    if (numBytes == 0) {
        return NULL_PTR;
    }

    // Use the header elements to track the length of allocated blocks.
    uint16_t i = 1;
    uint16_t limit = MEMORY_BYTES - numBytes - HEADER_SIZE + 1;
    Header *headerPtr;
    while (i < limit) {
        headerPtr = (Header *) (MEMORY + i);

        uint16_t allocSize = getAllocSize(*headerPtr);

        if (isFree(*headerPtr)) {
            // A zero-block has never been touched, so we can use it freely.
            if (allocSize == 0) {
                break;
            } else if (allocSize >= numBytes) {
                break;
            } else if (allocSize < numBytes) {
                // if the next block is zero, then there is "unlimited space" after this.
                Header *nextHeader = (Header *) (MEMORY + i + allocSize + HEADER_SIZE);
                if (isFree(*nextHeader) && getAllocSize(*nextHeader) == 0) {
                    break;
                }
            }
        }

        // If the current block is not free or large enough, then we
        // move to the next block.
        i += allocSize + HEADER_SIZE;
    }

    if (i >= limit) {
        return NULL_PTR;
    }

    // We have reached a free block. We allocate it using the header - data format.
    headerPtr = (Header *) (MEMORY + i);
    uint16_t prevNumBytes = getAllocSize(*headerPtr);
    
    *headerPtr = createHeader(numBytes, 1);

    // If needed, we split the current allocation into sections to preserve the free list structure.
    if (prevNumBytes - numBytes > 0) {
        // There are two possible cases here. (1) The new block is too small to fit a header and (2)
        // the new block is large enough to fit a header AND at least one data byte. We handle each case separately.
        if (prevNumBytes - numBytes <= HEADER_SIZE) {
            *headerPtr = setAllocSize(*headerPtr, prevNumBytes);  // Give the memory to this block (over-allocate)
        } else {
            uint16_t splitIndex = i + numBytes + HEADER_SIZE;
            Header *splitHeader = (Header *) (MEMORY + splitIndex);
            *splitHeader = createHeader(prevNumBytes - numBytes - HEADER_SIZE, 0);  // Remove the size of the new current header.
        }
    }

    return MEMORY + i + HEADER_SIZE;  // The data block starts one byte after the header.
}


void dealloc(void *ptr) {
    // The pointer points to the beginning of the data block. The header is located exactly one
    // byte behind this. We use this to mark the block as freed by setting the number of allocated.
    // bytes to zero.
    int8_t *bytePtr = (int8_t *) ptr;
    Header *headerPtr = (Header *) (bytePtr - HEADER_SIZE);

    uint16_t allocSize = getAllocSize(*headerPtr);
    
    // Coalesce the next blocks to reduce fragmentation.
    uint16_t offset = allocSize;
    Header *nextHeaderPtr = (Header *) (bytePtr + offset);
    while (isFree(*nextHeaderPtr) && getAllocSize(*nextHeaderPtr) > 0) {
        offset += HEADER_SIZE + getAllocSize(*nextHeaderPtr); // Add the header size to move beyond this block header
        nextHeaderPtr = (Header *) (bytePtr + offset);
    }

    if (offset > allocSize) {
        *headerPtr = setAllocSize(*headerPtr, offset);
    }

    // Mark this block as free
    *headerPtr = setAllocBit(*headerPtr, 0);
}


uint16_t allocBytes(void) {
    uint16_t bytes = 0;
    
    // Traverse the block headers to determine the number of allocated bytes
    int16_t i = 1;
    while (i < MEMORY_BYTES) {
        Header *headerPtr = (Header *) (MEMORY + i);
        uint16_t allocSize = getAllocSize(*headerPtr);

        if (allocSize == 0) {
            i += 1;
        } else {
            i += allocSize + HEADER_SIZE;
        }

        if (!isFree(*headerPtr)) {
            bytes += allocSize + HEADER_SIZE;
        }
    }

    return bytes;
}


uint16_t freeBytes(void) {
    return MEMORY_BYTES - allocBytes();
}


void clearMemory(void) {
    for (uint16_t i = 1; i < MEMORY_BYTES; i++) {
        MEMORY[i] = 0;
    }
}

