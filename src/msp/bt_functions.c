#include "bt_functions.h"

void send_char(char c) {
    // Wait until the Tx Buffer is free
    while(!(UCA3IFG & UCTXIFG));
    UCA3TXBUF = c;

    // Wait until byte has been sent.
    while(UCA3STATW & UCBUSY);
}

void send_message(char *str, uint16_t n) {
    uint16_t i = n;
    for (; i > 0; i--) {
        send_char(str[n - i]);
    }
}
