#include <msp430.h>

#include "DSPLib.h"
#include "neural_network_parameters.h"
#include "neural_network.h"
#include "math/matrix.h"
#include "init.h"
#include "bit_ops.h"
#include "adc.h"
#include "bt_functions.h"
#include "controller/pid_control.h"
#include "controller/budget_distribution.h"
#include "controller/interpolation.h"

#ifndef MAIN_GUARD
#define MAIN_GUARD

#define START_CHAR 'S'
#define DATA_CHAR 'D'
#define CONTROL_CHAR 'C'
#define PREDICTION_CHAR 'P'
#define SPACE ' '
#define MINUS '-'
#define NEWLINE '\n'
#define RESET_CHAR 'R'

#endif
