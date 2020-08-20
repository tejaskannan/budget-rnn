#include "main.h"


int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Not enough arguments. Must supply and input and output file for evaluation.\n");
        return 0;
    }

    char *inputs_path = argv[1];
    char *output_path = argv[2];

    FILE *inputs_file = fopen(inputs_path, "r");
    FILE *output_file = fopen(output_path, "r");
    int buffer_size = 500;
    char buffer[buffer_size];

    uint16_t outputBufferSize = 10;
    char outputBuffer[outputBufferSize];

    // Initialize an buffer for states
    matrix states[SEQ_LENGTH];
    matrix rnnResults[SEQ_LENGTH];

    int16_t stateData[SEQ_LENGTH][STATE_SIZE];
    for (uint16_t i = 0; i < SEQ_LENGTH; i++) {
        states[i].numRows = STATE_SIZE;
        states[i].numCols = 1;
        states[i].data = stateData[i];
    }

    // Initialize buffer of weights
    dtype weights[SEQ_LENGTH];

    // Create a zero state
    int16_t zeroData[STATE_SIZE];
    matrix zeroState;
    zeroState.numRows = STATE_SIZE;
    zeroState.numCols = 1;
    zeroState.data = zeroData;

    // Initialize an input buffer
    int16_t data[NUM_INPUT_FEATURES];
    matrix input;
    input.numRows = NUM_INPUT_FEATURES;
    input.numCols = 1;
    input.data = data;
    
    int16_t numSequences = SEQ_LENGTH / SAMPLES_PER_SEQ;

    int16_t output_buffer_size = 5;
    char output_buffer[output_buffer_size];
    
    int time = 0;

    uint16_t levelsToExecute;
    uint8_t isStopped;
    uint16_t currentLevel;
    uint8_t isEnd;

    uint16_t budgetIndex = 3;
    uint16_t numSamples = 0;
    uint16_t numCorrect = 0;
    uint16_t numLevels = 0;
    int16_t label;

    // Make a result buffer
    dtype resultData[STATE_SIZE];
    matrix resultBuffer;
    resultBuffer.numRows = STATE_SIZE;
    resultBuffer.numCols = 1;
    resultBuffer.data = resultData;
    matrix *result = &resultBuffer;

    // TODO: Add an avg power metric by computing the level distribution
    uint16_t levelCounts[numSequences];
    for (uint16_t i = 0; i < numSequences; i++) {
        levelCounts[i] = 0;
    }

    while (fgets(buffer, buffer_size, inputs_file) != NULL) {
     
        // Zero out the initial state
        matrix_set(&zeroState, 0);
   
        // Get the label
        fgets(outputBuffer, outputBufferSize, output_file);
        label = (int16_t) (outputBuffer[0] - '0');

        char *token = strtok(buffer, " ");

        // By default, we assume we are using one level
        levelsToExecute = 0;
        isStopped = 0;

        // Iterate through the sequence elements
        for (int16_t i = 0; i < SEQ_LENGTH; i++) {

            currentLevel = getCurrentLevel(i);

            // Fetch features for the i-th element
            for (int j = 0; j < NUM_INPUT_FEATURES; j++) {
                input.data[j] = atoi(token);
                token = strtok(NULL, " ");
            }

            // Don't process samples beyond the stopped level. In general, we wouldn't even capture such samples.
            // we only capture in this case because samples are read from a text file in a specific order.
            if (currentLevel > levelsToExecute && isStopped) {
                continue;
            }

            // Process the current input
            matrix currentState;
            if (i == 0) {
                currentState = zeroState;;
            } else {
                #ifdef IS_RNN
                    #ifdef IS_SAMPLE
                        matrix prevSampleState = (i - numSequences >= 0) ? states[i - numSequences] : zeroState;

                        // The first level does not use a fusion layer
                        if (currentLevel == 0) {
                            currentState = prevSampleState;
                        } else {
                            matrix prevLevelState = states[i-1];
                            fuse_states(&currentState, &prevSampleState, &prevLevelState, FIXED_POINT_PRECISION);
                        }
                    #else
                        currentState = states[i-1];
                    #endif
                #endif
            }

            apply_transformation(states + i, &input, &currentState, FIXED_POINT_PRECISION);

            // NBOW models use a weighted average pooling
            #ifndef IS_RNN
                weights[i] = compute_aggregation_weight(states + i, FIXED_POINT_PRECISION);
            #endif

            #ifdef IS_ADAPTIVE
                #ifdef IS_SAMPLE
                    if (i < numSequences) {
                #else
                    if ((i+1) % SAMPLES_PER_SEQ == 0) {
                #endif
                        // For NBOW SAMPLE models, we compute the stop output using a 'successive' state

                        #ifdef IS_RNN
                        int16_t stopProb = compute_stop_output(states + i, FIXED_POINT_PRECISION);
                        #else
                        pool_states(result, states, weights, i, 0, FIXED_POINT_PRECISION);
                        int16_t stopProb = compute_stop_output(result, FIXED_POINT_PRECISION);
                        #endif

                        int16_t threshold = THRESHOLDS[budgetIndex][currentLevel];

                        if (threshold < stopProb) {
                            levelsToExecute = currentLevel;
                            isStopped = 1;
                        }
                    }
            #endif

            // If we reach the last element of the highest level, we compute the output.
            isEnd = isLastSampleInLevel(i, SEQ_LENGTH);
            if ((currentLevel == levelsToExecute && isStopped && isEnd) || (i == SEQ_LENGTH - 1)) {
                #ifdef IS_RNN
                int16_t prediction = compute_prediction(states + i, FIXED_POINT_PRECISION);
                #else
                pool_states(result, states, weights, currentLevel, 1, FIXED_POINT_PRECISION);
                int16_t prediction = compute_prediction(result, FIXED_POINT_PRECISION);
                #endif

                numCorrect += (uint16_t) (prediction == label);
                numLevels += levelsToExecute + 1;
                levelCounts[levelsToExecute] += 1;
                break;
            }
        }

        time += 1;
        if (time % 1000 == 0) {
            printf("Finished %d samples\n", time);
        }
    }

    printf("Accuracy for model: %d / %d\n", numCorrect, time);
    printf("Average number of levels: %d / %d\n", numLevels, time);

    printf("{ ");
    for (uint16_t i = 0; i < numSequences; i++) {
        printf("%d ", levelCounts[i]);
    }
    printf("}\n");


    fclose(inputs_file);
    fclose(output_file);

    return 0;
}
