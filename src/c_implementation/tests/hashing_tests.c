#include "hashing_tests.h"

int main(void) {
    test_length_three();
    test_length_four();
    test_change_length();

    printf("Passed All Tests\n");
}

void test_length_three(void) {
    assert(pearson_hash("123", 3) == 7025);
    assert(pearson_hash("411", 3) == 34692);
    assert(pearson_hash("126", 3) == 18946);
    assert(pearson_hash("624", 3) == 34130);
}

void test_length_four(void) {
    assert(pearson_hash("123s", 4) == 41626);
    assert(pearson_hash("411s", 4) == 49222);
    assert(pearson_hash("126s", 4) == 17170);
    assert(pearson_hash("624s", 4) == 2460);
}

void test_change_length(void) {
    assert(pearson_hash("123s", 4) == 41626);
    assert(pearson_hash("123s", 3) == 7025);

    assert(pearson_hash("411s", 4) == 49222);
    assert(pearson_hash("411s", 3) == 34692);
}

