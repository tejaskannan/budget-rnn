#include <stdint.h>
#include "../math/fixed_point_ops.h"


#ifndef PID_CONTROL_GUARD
#define PID_CONTROL_GUARD

#define ERROR_QUEUE_SIZE 100

struct PidController {
    int16_t kp;
    int16_t ki;
    int16_t kd;
    uint16_t precision;
    int16_t errorQueue[ERROR_QUEUE_SIZE];  // Circular Buffer holding previous errors
    int16_t queuePos;
    int16_t integralWindow;
    int16_t integralMin;
    int16_t integralMax;
};

typedef struct PidController PidController;

void init_pid_controller(PidController *controller, uint16_t precision);
void add_error(int16_t error, PidController *controller);
int16_t control_step(int32_t y_true_lower, int32_t y_true_upper, int32_t y_pred, PidController *controller);

#endif
