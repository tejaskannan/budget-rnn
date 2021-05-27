#include <msp430.h>
#include "bit_ops.h"

#ifndef INIT_GUARD
#define INIT_GUARD

void init_gpio(void);
void init_uart_system(void);
void init_uart_pins(void);
void init_adc12(void);

#endif
