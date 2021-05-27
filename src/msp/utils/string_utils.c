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
    output[n] = '\0';  // Ensure the string is null-terminated

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


uint16_t append_int_to_str(char *output, uint16_t x) {
    int16_t digits[MAX_NUM_DIGITS];

    // As zero is the stopping condition, we handle it separately.
    if (x == 0) {
        output[0] = '0';
        output[1] = '\0';
        return 1;
    }

    int16_t i = 0;
    while (x) {
        digits[i++] = x % 10;
        x = x / 10;
    }

    uint16_t j;
    for (j = i; j > 0; j--) {
        output[j - 1] = (char) (digits[i - j] + '0');
    }
    output[i] = '\0';

    return i;
}
