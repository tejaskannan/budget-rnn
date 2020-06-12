#include "string_tests.h"

int main(void) {

    test_append_int();

    printf("Passed all tests.\n");

    return 0;
}


void test_append_int(void) {
    char str[5];
    str[0] = 't';

    int numDigits = append_int_to_str(str + 1, 31);
    assert(numDigits == 2);
    assert(strcmp(str, "t31") == 0);


    numDigits = append_int_to_str(str + 1, 1);
    assert(numDigits == 1);
    assert(strcmp(str, "t1") == 0);
}


