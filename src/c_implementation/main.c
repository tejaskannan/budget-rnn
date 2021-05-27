#include "main.h"

#define STEPS 100
#define BUDGET 9


int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Not enough arguments. Must supply and input and output file for evaluation.\n");
        return 0;
    }

    // Buffers for reading inputs
    char *inputs_path = argv[1];
    char *output_path = argv[2];

    FILE *inputs_file = fopen(inputs_path, "r");
    FILE *output_file = fopen(output_path, "r");
    int buffer_size = NUM_INPUT_FEATURES * SEQ_LENGTH * 6;
    char buffer[buffer_size];

    uint16_t outputBufferSize = 10;
    char outputBuffer[outputBufferSize];

    // Initialize an buffer for states
    matrix states[SEQ_LENGTH];

    int16_t stateData[SEQ_LENGTH * STATE_SIZE * VECTOR_COLS] = {0};
    for (uint16_t i = 0; i < SEQ_LENGTH; i++) {
        states[i].numRows = STATE_SIZE;
        states[i].numCols = VECTOR_COLS;
        states[i].data = &stateData[i * STATE_SIZE * VECTOR_COLS];
    }

    // Initialize a buffer for the logits
    matrix logits[NUM_OUTPUTS];
    int16_t logitsData[NUM_OUTPUTS * NUM_OUTPUT_FEATURES * VECTOR_COLS] = {0};
    for (uint16_t i = 0; i < NUM_OUTPUTS; i++) {
        logits[i].numRows = NUM_OUTPUT_FEATURES;
        logits[i].numCols = VECTOR_COLS;
        logits[i].data = &logitsData[i * NUM_OUTPUT_FEATURES * VECTOR_COLS];
    }

    // Initialize an input buffer
    int16_t data[NUM_INPUT_FEATURES * VECTOR_COLS];
    matrix input;
    input.numRows = NUM_INPUT_FEATURES;
    input.numCols = VECTOR_COLS;
    input.data = data;
    
    int16_t output_buffer_size = 5;
    char output_buffer[output_buffer_size];
    
    int time = 0;

    ExecutionState execState;

    // Track test statistics (useful for debugging)
    uint16_t numCorrect = 0;
    uint16_t numLevels = 0;
    int16_t label;

    uint16_t levelCounts[NUM_OUTPUTS];
    for (uint16_t i = 0; i < NUM_OUTPUTS; i++) {
        levelCounts[i] = 0;
    }

    int32_t energyBudget = int_to_fp32(BUDGET * STEPS, FIXED_POINT_PRECISION);
    int16_t thresholds[NUM_OUTPUTS] = { 0 };

    // Create the budget distribution
    #ifdef IS_BUDGET_RNN
    // Load the initial class counts
    int16_t budget = int_to_fp(BUDGET, FIXED_POINT_PRECISION);
    int32_t classCounts[NUM_OUTPUT_FEATURES][NUM_OUTPUTS];
    interpolate_counts(classCounts, budget, FIXED_POINT_PRECISION);

    BudgetDistribution distribution;
    init_distribution(&distribution, classCounts, STEPS, FIXED_POINT_PRECISION);

    // Create the PID controller
    PidController controller;
    init_pid_controller(&controller, FIXED_POINT_PRECISION);

    // Initialize the offset to the budget and get the initial thresholds
    int16_t budgetOffset = 0;
    interpolate_thresholds(thresholds, fp_add(budget, budgetOffset), FIXED_POINT_PRECISION);

    ConfidenceBound bound;
    bound.lower = 0;
    bound.upper = 0;
    #endif

    int16_t avgEnergy;
    int32_t totalEnergy = 0;
    int16_t stepEnergy = 0;
    uint8_t updateCounter = UPDATE_WINDOW;

    while (fgets(buffer, buffer_size, inputs_file) != NULL) {
     
        // Get the label
        fgets(outputBuffer, outputBufferSize, output_file);
        label = (int16_t) (outputBuffer[0] - '0');

        char *token = strtok(buffer, " ");

        // Initialize the execution state
        execState.levelsToExecute = 0;
        execState.isStopped = 0;
        execState.isCompleted = 0;
        execState.prediction = -1;
        execState.cumulativeUpdateProb = int_to_fp(1, FIXED_POINT_PRECISION);

        #ifdef IS_RNN
        execState.levelsToExecute = 7;
        execState.isStopped = 1;
        #endif

        // Iterate through the sequence elements
        uint16_t i;
        for (i = 0; i < SEQ_LENGTH; i++) {

            // Fetch features for the i-th element
            uint16_t j;
            for (j = 0; j < NUM_INPUT_FEATURES; j++) {
                input.data[j * VECTOR_COLS] = atoi(token);
                token = strtok(NULL, " ");
            }

            if (should_process(i, &execState)) {
                process_input(&input, states, logits, i, thresholds, &execState);
            } else {
                if (i > 0) {
                    matrix_replace(states + i, states + (i - 1));
                } else {
                    matrix_set(states + i, 0);
                }
            }

            #ifdef IS_SKIP_RNN
            // Update the state update probability
            int16_t nextUpdateProb = get_state_update_prob(states + i, execState.cumulativeUpdateProb, FIXED_POINT_PRECISION);
            execState.cumulativeUpdateProb = nextUpdateProb;
            #endif
        }

        time += 1;
        updateCounter -= 1;
        totalEnergy = fp32_add(totalEnergy, ENERGY_ESTIMATES[execState.levelsToExecute]);

        // printf("Prediction: %d\n", execState.prediction);
        numCorrect += (uint16_t) (execState.prediction == label);
        numLevels += execState.levelsToExecute + 1;

        #ifdef IS_BUDGET_RNN
        levelCounts[execState.levelsToExecute] += 1;

        if (updateCounter == 0) {
            bound = get_budget(energyBudget, time, STEPS, ENERGY_ESTIMATES, &distribution, FIXED_POINT_PRECISION);
        }

        avgEnergy = (int16_t) fp32_div(totalEnergy, int_to_fp32(time, FIXED_POINT_PRECISION), FIXED_POINT_PRECISION);
        budgetOffset = control_step(bound.lower, bound.upper, avgEnergy, &controller);

        // Update the distribution
        // stepEnergy = ENERGY_ESTIMATES[execState.levelsToExecute];
        // update_distribution(execState.prediction, execState.levelsToExecute, stepEnergy, &distribution, FIXED_POINT_PRECISION);
        // prevEnergy = ENERGY_ESTIMATES[execState.levelsToExecute];

        if (updateCounter == 0) {
            interpolate_thresholds(thresholds, fp_add(budget, budgetOffset), FIXED_POINT_PRECISION);
            updateCounter = UPDATE_WINDOW;
        }

        #endif

        if (time % 1000 == 0) {
            printf("Finished %d samples\n", time);
        }
    }

    printf("Accuracy for model: %d / %d\n", numCorrect, time);
    printf("Average number of levels: %d / %d\n", numLevels, time);
    
    float energyPerStep = ((float) totalEnergy) / ((1 << FIXED_POINT_PRECISION) * STEPS);
    printf("Average Energy per Step: %f\n", energyPerStep);
    
    printf("{ ");
    for (uint16_t i = 0; i < NUM_OUTPUTS; i++) {
        printf("%d ", levelCounts[i]);
    }
    printf("}\n");


    fclose(inputs_file);
    fclose(output_file);

    return 0;
}
