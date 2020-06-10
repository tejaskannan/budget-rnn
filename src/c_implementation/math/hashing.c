#include "hashing.h"


uint16_t pearson_hash(char *str, uint16_t n) {
    /**
     * Applies a 16-bit pearson hash function to the given string.
     */
    uint16_t elements[2];

    // Iteration variables
    uint8_t j;
    uint16_t i;

    // Stores the intermediate hash value
    uint8_t h;

    for (j = 0; j < 2; j++) {
        h = PERMUTE_TABLE[(((uint8_t) str[0]) + j) % 256];

        for (i = 1; i < n; i++) {
            h = PERMUTE_TABLE[h ^ ((uint8_t) str[i])];
        }

        elements[j] = h;
    }

    return (elements[0] << 8) | (elements[1]);
}
