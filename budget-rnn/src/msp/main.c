/**
 * Library to interact with an HM-10 Bluetooth Module.
 */
#include "main.h"

#define MAX_STEPS 100
#define BUDGET 9  // Energy Budget in mW. This is Total Budget / (sample rate * total budget * seq_length)
#define MILLI_FACTOR 1000
#define SAMPLE_RATE 2  // Sample period (1 / freq)

// Variable to track ADC conversion
volatile int16_t nAdc;

// Variable to track the inference thresholds
static int16_t thresholds[NUM_OUTPUTS];

// Variables to track energy consumption
volatile int32_t prevEnergy = 0;  // Previous energy in Joules (fixed-point)
volatile int32_t currentEnergy = 0;  // Current energy in Joules (fixed-point)
volatile int32_t avgPower = 0;  // Average power (fixed-point)
volatile int32_t totalEnergy = 0;  // Total consumed energy (fixed-point)
volatile int32_t stepEnergy = 0;  // Energy in the current step (fixed-point)
volatile int32_t stepPower = 0; // Power in mW over the last interval (fixed-point)
volatile int16_t steps = 0;

// Allocate memory for input features
static dtype inputData[NUM_INPUT_FEATURES * VECTOR_COLS];
static matrix inputFeatures;

// Allocate memory for logits
#pragma PERSISTENT(logitsData);
static dtype logitsData[SEQ_LENGTH * NUM_OUTPUT_FEATURES * VECTOR_COLS] = {0};
static matrix logits[SEQ_LENGTH];

// Allocate memory for hidden states
#pragma PERSISTENT(statesData);
static dtype statesData[SEQ_LENGTH * STATE_SIZE * VECTOR_COLS] = {0};
static matrix states[SEQ_LENGTH];

// Allocate memory for the response buffer
static char response[5];

// Variables for tracking the position in sequence, feature and inference
volatile uint16_t featureIdx = 0;
volatile uint16_t seqIdx = 1;
volatile int8_t sign = 1;
volatile ExecutionState execState;

void init_exec_state(void);


int main(void)
{
    WDTCTL = WDTPW | WDTHOLD;   // stop watchdog timer

    // Initialize the device UART settings
    init_uart_pins();

    // Initialize GPIO System
    init_gpio();

    // Disable the GPIO power-on default high-impedance mode to activate
    // previously configured port settings
    PM5CTL0 &= ~LOCKLPM5;

    // Initialize the clock and baudrate
    init_uart_system();

    // Initialize the ADC system
    init_adc12();

    // Assign memory for the states and logits
    uint16_t i = SEQ_LENGTH;
    for (; i > 0; i--) {
        states[i-1].numRows = STATE_SIZE;
        states[i-1].numCols = VECTOR_COLS;
        states[i-1].data = statesData + ((i - 1) * STATE_SIZE * VECTOR_COLS);

        logits[i-1].numRows = NUM_OUTPUT_FEATURES;
        logits[i-1].numCols = VECTOR_COLS;
        logits[i-1].data = logitsData + ((i - 1) * NUM_OUTPUT_FEATURES * VECTOR_COLS);
    }

    // Assign parameters of the input features matrix
    inputFeatures.numRows = NUM_INPUT_FEATURES;
    inputFeatures.numCols = VECTOR_COLS;
    inputFeatures.data = inputData;

    // Init the execution state
    init_exec_state();

    // Convert the budget to fixed point (mW)
    const int16_t budget = float_to_fp(BUDGET, FIXED_POINT_PRECISION);

    volatile int32_t energyBudget = ((int32_t) BUDGET) * ((int32_t) MAX_STEPS) * ((int32_t) SAMPLE_RATE) * ((int32_t) SEQ_LENGTH);  // Energy budget in mJ
    energyBudget = energyBudget / MILLI_FACTOR;  // Energy budget in Joules
    energyBudget = int_to_fp32(energyBudget, FIXED_POINT_PRECISION);

    #ifdef IS_BUDGET_RNN
    // Load the initial class counts
    int32_t classCounts[NUM_OUTPUT_FEATURES][NUM_OUTPUTS];
    interpolate_counts(classCounts, budget, FIXED_POINT_PRECISION);

    // Create the budget distribution
    BudgetDistribution distribution;
    init_distribution(&distribution, classCounts, MAX_STEPS, FIXED_POINT_PRECISION);

    // Create the PID controller
    PidController controller;
    init_pid_controller(&controller, FIXED_POINT_PRECISION);

    // Initialize the offset to the budget and get the initial thresholds
    volatile int16_t budgetOffset = 0;
    interpolate_thresholds(thresholds, budget, FIXED_POINT_PRECISION);

    volatile uint8_t updateCounter = UPDATE_WINDOW;

    volatile ConfidenceBound bound;
    bound.lower = -INT32_MAX;
    bound.upper = INT32_MAX;
    #endif

    #ifdef IS_SKIP_RNN
    int16_t oneHalf = 1 << (FIXED_POINT_PRECISION - 1);
    #endif

    volatile uint32_t cycleCount = 0;
    volatile uint32_t cycleCounter = 0;

    // Put into Low Power Mode
    __bis_SR_register(LPM0_bits | GIE);

    // Loop forever
    while (1) {

        // Incrementally execute the neural network, seqIdx is one beyond the current index (already incremented for the next iteration)
        process_input(&inputFeatures, states, logits, seqIdx - 1, thresholds, (ExecutionState *) &execState);

        #ifdef IS_SKIP_RNN
        // Update the state update probability
        int16_t nextUpdateProb = get_state_update_prob(states + seqIdx - 1, execState.cumulativeUpdateProb, FIXED_POINT_PRECISION);
        execState.cumulativeUpdateProb = nextUpdateProb;

        // Increment the sequence index according to the update probability. This prevents the system from collecting
        // uneeded samples. We stop when the update prob is greater than 0.5 or when we reach the end of the sequence.
        matrix *currentState = states + seqIdx - 1;
        while (seqIdx <= SEQ_LENGTH && execState.cumulativeUpdateProb <= oneHalf) {
            seqIdx += 1;
            execState.cumulativeUpdateProb = fp_add(execState.cumulativeUpdateProb, nextUpdateProb);
            matrix_replace(states + seqIdx - 1, currentState);
        }
        #endif

        if (seqIdx >= SEQ_LENGTH || (execState.prediction < NUM_OUTPUT_FEATURES && execState.isCompleted)) {

            // Use the ADC to get measure the remaining capacitor energy
            ADC12CTL0 |= ADC12ENC | ADC12SC;    // Start sampling/conversion
            __bis_SR_register(LPM0_bits | GIE);
            currentEnergy = get_energy(nAdc, FIXED_POINT_PRECISION);  // Energy in J

            // Get the energy for the just-completed step
            stepEnergy = fp32_sub(prevEnergy, currentEnergy);
            if (stepEnergy < 0) {
                stepEnergy = 0;
            }

            // Add to the the total consumed energy thus far (in Joules)
            totalEnergy = fp32_add(totalEnergy, stepEnergy);

            response[0] = PREDICTION_CHAR;
            response[1] = execState.prediction;
            response[2] = (char) ((nAdc >> 8) & 0xFF);
            response[3] = (char) (nAdc & 0xFF);

            send_message(response, 4);

            steps++;
            prevEnergy = currentEnergy;

            #ifdef IS_BUDGET_RNN

            updateCounter--;

            // Update the budget distribution using the results of the current inference
            stepPower = (stepEnergy * MILLI_FACTOR) / (SEQ_LENGTH * SAMPLE_RATE);  // Approximate power this step in mW
            update_distribution(execState.prediction, execState.levelsToExecute, stepPower, &distribution, FIXED_POINT_PRECISION);

            if (updateCounter == 0) {
                bound = get_budget(energyBudget, steps, MAX_STEPS, ENERGY_ESTIMATES, &distribution, FIXED_POINT_PRECISION);
            }

            // Compute the average energy per step in mW
            avgPower = (totalEnergy * MILLI_FACTOR) / (steps * SAMPLE_RATE * SEQ_LENGTH);

            budgetOffset = control_step(bound.lower, bound.upper, avgPower, &controller);

            if (updateCounter == 0) {
                interpolate_thresholds(thresholds, fp_add(budget, budgetOffset), FIXED_POINT_PRECISION); // Interpolate the new thresholds
                updateCounter = UPDATE_WINDOW;  // Reset the update counter
            }
            #endif

            // Reset the execution state
            init_exec_state();
        } else {
            response[0] = CONTROL_CHAR;
            response[1] = NUM_OUTPUTS;
            response[2] = STRIDE_LENGTH;

            // The exec field changes based on the model type. Different
            // models use different techniques to skip inputs.
            #ifdef IS_BUDGET_RNN
            response[3] = execState.levelsToExecute;
            #elif defined(IS_SKIP_RNN)
            response[3] = seqIdx;
            #endif

            send_message(response, 4);
        }

        // Put back into LPM
        __bis_SR_register(LPM0_bits | GIE);
    }

    return 0;
}


void init_exec_state(void) {
    // Initialize execution state
    execState.budgetIndex = 0;
    execState.levelsToExecute = NUM_OUTPUTS;
    execState.isStopped = 0;
    execState.isCompleted = 0;
    execState.cumulativeUpdateProb = 0;
    execState.prediction = NUM_OUTPUT_FEATURES;

    #ifdef IS_RNN
    execState.isStopped = 1;
    #endif
}


/**
 * ISR for receiving data on UART RX pin.
 */
#pragma vector=EUSCI_A3_VECTOR
__interrupt void USCI_A3_ISR(void) {
    char c;

    switch(__even_in_range(UCA3IV, USCI_UART_UCTXCPTIFG)) {
        case USCI_NONE: break;
        case USCI_UART_UCRXIFG:
            // Wait until TX Buffer is not busy
            while(!TEST_BIT(UCA3IFG, UCTXIFG));

            c = (char) UCA3RXBUF;

            if (c == START_CHAR || c == DATA_CHAR) {
                // For either the start or data character, we zero-out the feature vector.
                if (c == START_CHAR) {
                    seqIdx = 0;
                }

                featureIdx = 0;
                sign = 1;

                uint16_t i = NUM_INPUT_FEATURES * VECTOR_COLS;
                for (; i > 0; i--) {
                    inputFeatures.data[i - 1] = 0;
                }
            } else if (c == RESET_CHAR) {
                prevEnergy = 0;
                currentEnergy = 0;
                avgPower = 0;
                totalEnergy = 0;
                stepEnergy = 0;
                steps = 0;
            } else if (c == SPACE || c == NEWLINE) {
                // A space indicates that the current feature has ended. We thus
                // save the current feature value and the advance the feature index.
                // If we reach the end of the feature vector, we exit low power mode
                // and process the feature.
                inputFeatures.data[featureIdx * VECTOR_COLS] *= sign;

                // Initialize the next feature value
                featureIdx += 1;
                sign = 1;

                if (c == NEWLINE && featureIdx >= NUM_INPUT_FEATURES) {
                    #if defined(IS_BUDGET_RNN)
                    // Advance the sequence index counter to the next sample based on the number
                    // of levels to execute
                    if (STRIDE_LENGTH > 1) {
                        uint16_t currentLevel = seqIdx % NUM_OUTPUTS;

                        if (currentLevel > execState.levelsToExecute) {
                            seqIdx += NUM_OUTPUTS - currentLevel;
                        }
                    }
                    #endif

                    seqIdx += 1;

                    __bic_SR_register_on_exit(LPM0_bits | GIE);
                }
            } else if (c == MINUS) {
                // If we encounter a minus sign, we set the sign to -1. This
                // factor is applied once the entire feature is collected.
                sign = -1;
            } else if (c >= '0' && c <= '9') {
                // Numerical digits are saved into the corresponding features.
                // We assume that features are send in a decimal, big-endian format
                // (i.e. most significant digits sent first).
                dtype currentValue = inputFeatures.data[featureIdx * VECTOR_COLS];
                inputFeatures.data[featureIdx * VECTOR_COLS] = currentValue * 10 + (c - '0');
            }

            break;
        case USCI_UART_UCTXIFG: break;
        case USCI_UART_UCSTTIFG: break;
        case USCI_UART_UCTXCPTIFG: break;
        default: break;
    }
}

#pragma vector = ADC12_B_VECTOR
__interrupt void ADC12_ISR(void)
{
    switch(__even_in_range(ADC12IV, ADC12IV__ADC12RDYIFG))
    {
        case ADC12IV__NONE:        break;   // Vector  0:  No interrupt
        case ADC12IV__ADC12OVIFG:  break;   // Vector  2:  ADC12MEMx Overflow
        case ADC12IV__ADC12TOVIFG: break;   // Vector  4:  Conversion time overflow
        case ADC12IV__ADC12HIIFG:  break;   // Vector  6:  ADC12BHI
        case ADC12IV__ADC12LOIFG:  break;   // Vector  8:  ADC12BLO
        case ADC12IV__ADC12INIFG:  break;   // Vector 10:  ADC12BIN
        case ADC12IV__ADC12IFG0:            // Vector 12:  ADC12MEM0 Interrupt
            nAdc = (int16_t) ADC12MEM0;

            // Exit from LPM0 and continue executing main
            __bic_SR_register_on_exit(LPM0_bits);
            break;
        case ADC12IV__ADC12IFG1:   break;   // Vector 14:  ADC12MEM1
        case ADC12IV__ADC12IFG2:   break;   // Vector 16:  ADC12MEM2
        case ADC12IV__ADC12IFG3:   break;   // Vector 18:  ADC12MEM3
        case ADC12IV__ADC12IFG4:   break;   // Vector 20:  ADC12MEM4
        case ADC12IV__ADC12IFG5:   break;   // Vector 22:  ADC12MEM5
        case ADC12IV__ADC12IFG6:   break;   // Vector 24:  ADC12MEM6
        case ADC12IV__ADC12IFG7:   break;   // Vector 26:  ADC12MEM7
        case ADC12IV__ADC12IFG8:   break;   // Vector 28:  ADC12MEM8
        case ADC12IV__ADC12IFG9:   break;   // Vector 30:  ADC12MEM9
        case ADC12IV__ADC12IFG10:  break;   // Vector 32:  ADC12MEM10
        case ADC12IV__ADC12IFG11:  break;   // Vector 34:  ADC12MEM11
        case ADC12IV__ADC12IFG12:  break;   // Vector 36:  ADC12MEM12
        case ADC12IV__ADC12IFG13:  break;   // Vector 38:  ADC12MEM13
        case ADC12IV__ADC12IFG14:  break;   // Vector 40:  ADC12MEM14
        case ADC12IV__ADC12IFG15:  break;   // Vector 42:  ADC12MEM15
        case ADC12IV__ADC12IFG16:  break;   // Vector 44:  ADC12MEM16
        case ADC12IV__ADC12IFG17:  break;   // Vector 46:  ADC12MEM17
        case ADC12IV__ADC12IFG18:  break;   // Vector 48:  ADC12MEM18
        case ADC12IV__ADC12IFG19:  break;   // Vector 50:  ADC12MEM19
        case ADC12IV__ADC12IFG20:  break;   // Vector 52:  ADC12MEM20
        case ADC12IV__ADC12IFG21:  break;   // Vector 54:  ADC12MEM21
        case ADC12IV__ADC12IFG22:  break;   // Vector 56:  ADC12MEM22
        case ADC12IV__ADC12IFG23:  break;   // Vector 58:  ADC12MEM23
        case ADC12IV__ADC12IFG24:  break;   // Vector 60:  ADC12MEM24
        case ADC12IV__ADC12IFG25:  break;   // Vector 62:  ADC12MEM25
        case ADC12IV__ADC12IFG26:  break;   // Vector 64:  ADC12MEM26
        case ADC12IV__ADC12IFG27:  break;   // Vector 66:  ADC12MEM27
        case ADC12IV__ADC12IFG28:  break;   // Vector 68:  ADC12MEM28
        case ADC12IV__ADC12IFG29:  break;   // Vector 70:  ADC12MEM29
        case ADC12IV__ADC12IFG30:  break;   // Vector 72:  ADC12MEM30
        case ADC12IV__ADC12IFG31:  break;   // Vector 74:  ADC12MEM31
        case ADC12IV__ADC12RDYIFG: break;   // Vector 76:  ADC12RDY
        default: break;
    }
}

