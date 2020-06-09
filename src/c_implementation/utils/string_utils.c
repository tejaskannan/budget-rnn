#include "string_utils.h"


uint16_t string_length(char *str) {
    /**
     * Computes the length of a null-terminated string.
     */
    uint16_t n = 0;

    while (str[n] && n < MAX_STR_LENGTH) {
        n++;
    }

    return n;
}

char *string_copy(char *output, char *str, uint16_t n) {
    for (; n > 0; n--) {
        output[n - 1] = str[n - 1];
    }

    return output;
}

char *replace(char *output, const char *str, uint16_t start) {
    uint16_t i = 0;
    while (str[i]) {
        output[start + i] = str[i];
        i++;
    }

    return output;
}
