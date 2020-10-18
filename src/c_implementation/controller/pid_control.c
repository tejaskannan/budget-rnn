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
}


void add_error(int16_t error, PidController *controller) {
    controller->errorQueue[controller->queuePos] = error;

    controller->queuePos += 1;
    if (controller->queuePos >= ERROR_QUEUE_SIZE) {
        controller->queuePos = 0;
    }
}


int16_t prev_error(PidController *controller, uint16_t pos) {
    uint16_t prevIndex;
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
    int16_t prevError, error;
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


int16_t control_step(int16_t y_true_lower, int16_t y_true_upper, int16_t y_pred, PidController *controller) {
    /**
     * Runs the controller for one iteration and returns the next output.
     *
     * Args:
     *  y_true: The reference value (a standard integer)
     *  y_pred: The prediction value (a standard integer)
     *  time: The current time step
     *  controller: Information of the PID controller.
     *
     * Returns:
     *  The prediction for the next iteration. This is a standard integer.
     */
    // Create the error signal
    int16_t error;
    if (y_true_lower <= y_pred && y_pred <= y_true_upper) {
        error = 0;
    } else if (y_pred < y_true_lower) {
        error = fp_sub(y_true_lower, y_pred);
    } else {
        error = fp_sub(y_true_upper, y_pred);
    }

    // Add error to the error queue
    add_error(error, controller);

    int16_t integral = integral_error(controller);
    int16_t integral_term = fp_mul(controller->ki, integral, controller->precision);

    int16_t proportional_term = fp_mul(controller->kp, error, controller->precision);

    // TODO: Compute derivative term

    int16_t control_error = fp_add(integral_term, proportional_term);

    if (error == 0) {
        return 0;
    }
    return control_error; 

//    // Convert the true and reference values to fixed point
//    y_true = int_to_fp(y_true, controller->precision);
//    y_pred = int_to_fp(y_pred, controller->precision);
//
//
//
//
//    // Calculate the error
//    int16_t error = fp_add(y_true, fp_neg(y_pred));
//
//    // Calculate the proportional error
//    int16_t prop_error = fp_mul(error, controller->kp, controller->precision);
//
//    // Calculate the (approximate) integral error
//    int16_t time_delta = fp_add(time, fp_neg(controller->prev_time));
//    int16_t height = fp_div(fp_add(error, controller->prev_error), int_to_fp(2, controller->precision), controller->precision);
//    int16_t integral_error = fp_mul(time_delta, height, controller->precision);
//    integral_error = fp_mul(integral_error, controller->ki, controller->precision);
//    integral_error = fp_add(integral_error, controller->integral);
//
//    // Compute the control signal
//    int16_t control_signal = fp_add(prop_error, integral_error);
//
//    // Update the controller
//    controller->integral = integral_error;
//    controller->prev_error = error;
//    controller->prev_time = time;
//
//    // Apply the process control function
//    int16_t next_pred = fp_add(y_pred, control_signal);
//    next_pred = fp_round_to_int(next_pred, controller->precision);
//
//    // Convert back to an integer
//    int16_t rounded_pred = (int16_t) (next_pred >> controller->precision);
//    if (rounded_pred > controller->max_value) {
//        return controller->max_value;
//    } else if (rounded_pred < controller->min_value) {
//        return controller->min_value;
//    }
//    return rounded_pred;
}
