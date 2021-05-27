#include <stdint.h>
#include "math/fixed_point_ops.h"

#ifndef ADC_HEADER
#define ADC_HEADER

    void init_power_monitor(uint16_t precision);
    int32_t get_voltage(int16_t nAdc);
    int32_t get_energy(int16_t nAdc, uint16_t precision);

#endif
