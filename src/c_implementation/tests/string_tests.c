#include "string_tests.h"

int main(void) {

    test_append_int();
    test_replace();
    test_copy();
    test_length();

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


void test_length(void) {
    char str1[6] = "hello";
    char str2[3] = "10";


    assert(5 == string_length(str1));
    assert(2 == string_length(str2));
}


void test_copy(void) {
    char str1[6] = "hello";
    char str[6];

    string_copy(str, str1, 6);
    assert(strcmp(str, "hello") == 0);
    
    string_copy(str, str1, 3);
    assert(strcmp(str, "hel") == 0);
}


void test_replace(void) {
    char str[6] = "abcd";

    replace(str, "oh", 0);
    assert(strcmp(str, "ohcd") == 0);

    replace(str, "io", 2);    
    assert(strcmp(str, "ohio") == 0);
}
