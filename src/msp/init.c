/*
 * This file contains functions which initialize the device. This initialization
 * includes MSP pins, UART configuration, and the clock system.
 */
#include "init.h"

void init_gpio(void) {
    /**
     * Initializes all pins to output and sets pins to LOW. This
     * prevents unnecessary current consumption by floating pins.
     */
    P1DIR = 0xFF;
    P1OUT = 0x0;
    P2DIR = 0xFF;
    P2OUT = 0x0;
    P3DIR = 0xFF;
    P3OUT = 0x0;
    P4DIR = 0xFF;
    P4OUT = 0x0;
    P5DIR = 0xFF;
    P5OUT = 0x0;
    P6DIR = 0xFF;
    P6OUT = 0x0;
    P7DIR = 0xFF;
    P7OUT = 0x0;
    P8DIR = 0xFF;
    P8OUT = 0x0;
    PADIR = 0xFF;
    PAOUT = 0x0;
    PBDIR = 0xFF;
    PBOUT = 0x0;
    PCDIR = 0xFF;
    PCOUT = 0x0;
    PDDIR = 0xFF;
    PDOUT = 0x0;
}

void init_uart_pins(void) {
    /*
     * Configures USCI_A3 Pins
     */
    CLR_BIT(P6SEL1, (BIT0 | BIT1));
    SET_BIT(P6SEL0, (BIT0 | BIT1));
}

void init_uart_system(void) {
    /**
     * Initializes the UART system by setting the correct baudrate.
     */
    // Set clock system with DCO of ~1MHz
    CSCTL0_H = CSKEY_H;  // Unlock clock system control registers
    CSCTL1 = DCOFSEL_0;  // Set DCO to 1MHz
    CSCTL2 = SELS__DCOCLK | SELM__DCOCLK;
    CSCTL3 =  DIVA__1 | DIVS__1 | DIVM__1;  // Set dividers
    CSCTL0_H = 0;   // Lock the clock system control registers

    // Configure USCI_A3 for UART
    UCA3CTLW0 = UCSWRST;  // Put eUSCI in reset
    UCA3CTLW0 |= UCSSEL__SMCLK;  // CLK = SMCLK
    UCA3BRW = 6;  // integer part of (10^6) / (16 * 9600)
    UCA3MCTLW |= UCOS16 | UCBRF_8 | 0xAA;  // UCBRSx = 0xAA (User Guide Table 30-4)
    UCA3CTLW0 &= ~UCSWRST;  // Initialize eUSCI
    UCA3IE |= UCRXIE;  // Enable RX Interrupt
}

void init_adc12(void) {
    P1SEL1 |= BIT2;                         // Configure P1.2 for ADC
    P1SEL0 |= BIT2;

    while(REFCTL0 & REFGENBUSY);            // If ref generator busy, WAIT
    REFCTL0 |= REFVSEL_2 | REFON;           // Select internal ref = 2.5V
                                            // Internal Reference ON

    // Configure ADC12
    ADC12CTL0 = ADC12SHT0_2 | ADC12ON;
    ADC12CTL1 = ADC12SHP;                   // ADCCLK = MODOSC; sampling timer
    ADC12CTL2 |= ADC12RES_2;                // 12-bit conversion results
    ADC12IER0 |= ADC12IE0;                  // Enable ADC conv complete interrupt
    ADC12MCTL0 |= ADC12INCH_2 | ADC12VRSEL_1; // A2 ADC input select; Vref = 2.5V

    while(!(REFCTL0 & REFGENRDY));          // Wait for reference generator
                                            // to settle
}
