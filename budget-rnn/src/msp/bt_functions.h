#include <msp430.h>
#include <stdint.h>

#ifndef BT_FUNCTIONS_GUARD
#define BT_FUNCTIONS_GUARD

void send_char(char c);
void send_message(char *str, uint16_t n);

#endif
