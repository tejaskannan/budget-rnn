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

   // char *token = strtok(buffer, " ");
   // while (token != NULL) {
   //     printf("%s\n", token);
   //     token = strtok(NULL, " ");
   // }

    // Initialize a dummy input array
    // matrix **inputs = (matrix **) alloc(sizeof(matrix *) * SEQ_LENGTH);
    matrix *inputs[SEQ_LENGTH];
    for (int16_t i = 0; i < SEQ_LENGTH; i++) {
        inputs[i] = matrix_allocate(NUM_INPUT_FEATURES, 1);
    }
    
    int16_t output_buffer_size = 5;
    char output_buffer[output_buffer_size];

    int16_t num_correct = 0;
    int16_t num_samples = 0;
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

        uint16_t prediction = execute_model(inputs);
        
        fgets(output_buffer, output_buffer_size, output_file);
        int16_t label = atoi(output_buffer);

        num_correct += (int16_t) (label == prediction);
        num_samples += 1;
    }

    printf("Accuracy: %d / %d\n", num_correct, num_samples);
    fclose(inputs_file);
    fclose(output_file);

    // Release all memory
    for (int16_t i = 0; i < SEQ_LENGTH; i++) {
        matrix_free(inputs[i]);
    }

    return 0;
}

