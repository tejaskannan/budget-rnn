#include "main.h"


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

    // TODO: Add an avg power metric by computing the level distribution
    uint16_t levelCounts[NUM_OUTPUTS];
    for (uint16_t i = 0; i < NUM_OUTPUTS; i++) {
        levelCounts[i] = 0;
    }

    while (fgets(buffer, buffer_size, inputs_file) != NULL) {
     
        // Get the label
        fgets(outputBuffer, outputBufferSize, output_file);
        label = (int16_t) (outputBuffer[0] - '0');

        char *token = strtok(buffer, " ");

        // Initialize the execution state
        execState.budgetIndex = 0;
        execState.levelsToExecute = 0;
        execState.isStopped = 0;
        execState.isCompleted = 0;
        execState.prediction = -1;
        execState.cumulativeUpdateProb = int_to_fp(1, FIXED_POINT_PRECISION);

        #ifdef IS_RNN
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
                process_input(&input, states, i, &execState);
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

        numCorrect += (uint16_t) (execState.prediction == label);
        numLevels += execState.levelsToExecute + 1;
        levelCounts[execState.levelsToExecute] += 1;

        time += 1;
        if (time % 1000 == 0) {
            printf("Finished %d samples\n", time);
        }
    }

    printf("Accuracy for model: %d / %d\n", numCorrect, time);
    printf("Average number of levels: %d / %d\n", numLevels, time);

    printf("{ ");
    for (uint16_t i = 0; i < NUM_OUTPUTS; i++) {
        printf("%d ", levelCounts[i]);
    }
    printf("}\n");


    fclose(inputs_file);
    fclose(output_file);

    return 0;
}
