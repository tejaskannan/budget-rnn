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

    // Initialize a dummy input array
    matrix *inputs[SEQ_LENGTH];
    for (int16_t i = 0; i < SEQ_LENGTH; i++) {
        inputs[i] = matrix_allocate(NUM_INPUT_FEATURES, 1);
    }
    
    int16_t num_sequences = SEQ_LENGTH / SAMPLES_PER_SEQ;

    int16_t output_buffer_size = 5;
    char output_buffer[output_buffer_size];
    // int8_t outputs[num_sequences];
    InferenceResult result;

    // int16_t num_correct[num_sequences];
    int16_t num_samples = 0;
    //for (int16_t i = 0; i < num_sequences; i++) {
    //    num_correct[i] = 0;
    //}
    int16_t num_correct = 0;
    int levels = 0;

    while (fgets(buffer, buffer_size, inputs_file) != NULL) {
        char *token = strtok(buffer, " ");
        for (int16_t i = 0; i < SEQ_LENGTH; i++) {
            for (int j = 0; j < NUM_INPUT_FEATURES; j++) {
                float feature_val = atof(token);
                inputs[i]->data[j] = float_to_fp(feature_val, FIXED_POINT_PRECISION);

                token = strtok(NULL, " ");
            }

            normalize(inputs[i], INPUT_MEAN, INPUT_STD, FIXED_POINT_PRECISION);
        }

        execute_model(inputs, &result);

        fgets(output_buffer, output_buffer_size, output_file);
        int16_t label = atoi(output_buffer);

        if (result.prediction == label) {
            num_correct += 1;
        }

        levels += result.num_levels;

        //for (int16_t i = 0; i < num_sequences; i++) {
        //    if (label == outputs[i]) {
        //        num_correct[i] += 1;
        //    }
        //}
        num_samples += 1;

        if (num_samples % 1000 == 0) {
            printf("Finished %d samples\n", num_samples);
        }
    }

    printf("Accuracy for model: %d / %d\n", num_correct, num_samples);
    printf("Average number of levels: %d / %d\n", levels, num_samples);

    //for (int16_t i = 0; i < num_sequences; i++) {
    //    printf("Accuracy for level %d: %d / %d\n", i + 1, num_correct[i], num_samples);
    //}

    fclose(inputs_file);
    fclose(output_file);

    // Release all memory
    for (int16_t i = 0; i < SEQ_LENGTH; i++) {
        matrix_free(inputs[i]);
    }

    return 0;
}
