#include "adc.h"


#define HALF_CAPACITANCE 2


int32_t get_voltage(int16_t nAdc) {
    return  2 * ((((int32_t) nAdc) * 2500L) >> 12);  // Units in mV
}

int32_t get_energy(int16_t nAdc, uint16_t precision) {
    int32_t voltage = get_voltage(nAdc);  // Standard integer, units mV
    voltage = int_to_fp32(voltage, precision) / 1000;  // Fixed-point integer, units V
    int32_t vSquared = fp32_mul(voltage, voltage, precision);  // Standard integer, units (V)^2
    return ((int32_t) HALF_CAPACITANCE) * vSquared;
}
