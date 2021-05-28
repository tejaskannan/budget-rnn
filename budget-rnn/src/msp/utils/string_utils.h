/**
 * Collection of string functions. These are mainly useful for handling seed values.
 */

#include <stdint.h>


#ifndef STRING_UTILS_GUARD
#define STRING_UTILS_GUARD

    #define MAX_STR_LENGTH 10000
    #define MAX_NUM_DIGITS 10

    uint16_t string_length(char *str);
    char *string_copy(char *output, char *str, uint16_t n);
    char *replace(char *output, const char *str, uint16_t start);
    uint16_t append_int_to_str(char *output, uint16_t x);

#endif

