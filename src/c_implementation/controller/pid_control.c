#include "pid_control.h"


void init_pid_controller(PidController *controller, uint16_t precision) {
    controller->kp = 1 << precision;  // 1
    controller->kd = 1 << (precision - 5);  // 1 / 32
    controller->ki = 1 << (precision - 3);  // 1 / 8
    controller->precision = precision;
    controller->queuePos = 0;
    controller->integralWindow = 80;
    controller->integralMin = int_to_fp(-16, precision);
    controller->integralMax = int_to_fp(16, precision);

    uint16_t i = ERROR_QUEUE_SIZE;
    for (; i > 0; i--) {
        controller->errorQueue[i] = 0;
    }
}


void add_error(int16_t error, PidController *controller) {
    controller->errorQueue[controller->queuePos] = error;

    controller->queuePos += 1;
    if (controller->queuePos >= ERROR_QUEUE_SIZE) {
        controller->queuePos = 0;
    }
}


int16_t prev_error(PidController *controller, uint16_t pos) {
    uint16_t prevIndex = 0;
    if (pos > 0) {
        prevIndex = pos - 1;
    } else {
        prevIndex = ERROR_QUEUE_SIZE - 1;
    }

    return controller->errorQueue[prevIndex];
}


int16_t integral_error(PidController *controller) {
    uint16_t windowIdx = controller->integralWindow;
    uint16_t queueIdx = controller->queuePos;
    int16_t prevError = 0;
    int16_t error = 0;
    int16_t integral = 0;

    for (; windowIdx > 0; windowIdx--) {
        prevError = prev_error(controller, queueIdx);
        error = controller->errorQueue[queueIdx];
        integral = fp_add(integral, fp_sub(error, prevError));
        
        if (integral > controller->integralMax) {
            return controller->integralMax;
        } else if (integral < controller->integralMin) {
            return controller->integralMin;
        }

        if (queueIdx == 0) {
            queueIdx = ERROR_QUEUE_SIZE - 1;
        } else {
            queueIdx -= 1;
        }
    }

    return integral;
}


int16_t control_step(int32_t y_true_lower, int32_t y_true_upper, int32_t y_pred, PidController *controller) {
    /**
     * Runs the controller for one iteration and returns the next output.
     *
     * Args:
     *  y_true_lower: The reference value lower bound (a fixed-point integer)
     *  y_true_upper: The reference value upper bound (a fixed-point integer)
     *  y_pred: The prediction value (a fixed-point integer)
     *  controller: The PID controller.
     *
     * Returns:
     *  The prediction for the next iteration. This is a standard integer.
     */

    // Create the error signal
    int32_t error = 0;
    if (y_true_lower <= y_pred && y_pred <= y_true_upper) {
        error = 0;
    } else if (y_pred < y_true_lower) {
        error = fp32_sub(y_true_lower, y_pred);
    } else {
        error = fp32_sub(y_true_upper, y_pred);
    }

    int16_t clippedError = 0;
    if (error >= INT16_MAX) {
        clippedError = INT16_MAX;
    } else if (error <= -INT16_MAX) {
        clippedError = -INT16_MAX;
    } else {
        clippedError = (int16_t) error;
    }

    // Add error to the error queue
    add_error(clippedError, controller);

    int16_t integral = integral_error(controller);
    int16_t integral_term = fp_mul(controller->ki, integral, controller->precision);

    int16_t proportional_term = fp_mul(controller->kp, clippedError, controller->precision);

    int16_t control_error = fp_add(integral_term, proportional_term);

    if (error == 0) {
        return 0;
    }
    return control_error;
}
