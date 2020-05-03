#include "memory_tests.h"


int main(void) {

    test_alloc_free();
    test_store_byte();
    test_pair();
    test_triple();
    test_struct();

    printf("Passed All Tests\n\n");
    return 0;
}


void test_alloc_free(void) {
    int8_t numBytes = 16;
    int8_t *ptr = alloc(numBytes);
    
    assert(numBytes + 1 == allocBytes());

    free(ptr);
    assert(0 == allocBytes());
}


void test_store_byte(void) {
    int8_t x = 10;
    int8_t *ptr = alloc(1);
    assert(2 == allocBytes());

    // Set the value
    *ptr = x;

    assert(x == *ptr);
    
    free(ptr);
    assert(0 == allocBytes());
    assert(0 == *ptr);
}


void test_pair(void) {
    int8_t numBytesOne = 2;
    int8_t x1 = 10;
    int8_t *ptr1 = alloc(numBytesOne);
    *ptr1 = x1;
    assert(x1 == *ptr1);
    assert(numBytesOne + 1 == allocBytes());

    int8_t numBytesTwo = 4;
    int8_t x2 = 100;
    int8_t *ptr2 = alloc(numBytesTwo);
    *ptr2 = x2;
    assert(x2 == *ptr2);
    assert(numBytesOne + numBytesTwo + 2 == allocBytes());

    free(ptr1);
    assert(numBytesTwo + 1 == allocBytes());
    assert(0 == *ptr1);
    assert(x2 == *ptr2);

    free(ptr2);
    assert(0 == allocBytes());
    assert(0 == *ptr2);
}


void test_triple(void) {
    int8_t numBytesOne = 1;
    int8_t x1 = 10;
    int8_t *ptr1 = alloc(numBytesOne);
    *ptr1 = x1;
    assert(x1 == *ptr1);
    assert(numBytesOne + 1 == allocBytes());

    int8_t numBytesTwo = 4;
    int8_t x2 = 100;
    int8_t *ptr2 = alloc(numBytesTwo);
    *ptr2 = x2;
    assert(x2 == *ptr2);
    assert(numBytesOne + numBytesTwo + 2 == allocBytes());

    int8_t numBytesThree = 6;
    int8_t x3 = 50;
    int8_t *ptr3 = alloc(numBytesThree);
    *ptr3 = x3;

    assert(x3 == *ptr3);
    assert(numBytesOne + numBytesTwo + numBytesThree + 3 == allocBytes());

    // Free the middle pointer and make sure everything else is still present
    free(ptr2);
    assert(x1 == *ptr1);
    assert(0 == *ptr2);
    assert(x3 == *ptr3);
    assert(numBytesOne + numBytesThree + 2 == allocBytes());

    free(ptr1);
    free(ptr3);
    assert(0 == allocBytes());
}


void test_struct(void) {
    int16_t x = 5;
    int16_t y = 1000;

    Point *ptr = (Point *) alloc(sizeof(Point));
    ptr->x = x;
    ptr->y = y;

    assert(sizeof(Point) + 1 == allocBytes());
    assert(x == ptr->x);
    assert(y == ptr->y);

    free(ptr);
    assert(0 == allocBytes());
}


void test_struct_multiple(void) {
    int16_t x1 = 5000;
    int16_t y1 = 1000;
    Point *ptr1 = (Point *) alloc(sizeof(Point));
    ptr1->x = x1;
    ptr1->y = y1;

    assert(sizeof(Point) + 1 == allocBytes());
    assert(x1 == ptr1->x);
    assert(y1 == ptr1->y);

    int16_t x2 = -1000;
    int16_t y2 = -2500;
    Point *ptr2 = (Point *) alloc(sizeof(Point));
    ptr2->x = x2;
    ptr2->y = y2;

    assert(2 * (sizeof(Point) + 1) == allocBytes());
    assert(x2 == ptr2->x);
    assert(y2 == ptr2->y);

    free(ptr1);
    assert(sizeof(Point) + 1 == allocBytes());
    assert(x2 == ptr2->x);
    assert(y2 == ptr2->y);

    free(ptr2);
    assert(0 == allocBytes);
}
