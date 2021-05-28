# C Neural Network Implementation
This folder contains a C implementation of the RNN models for standard systems. It is mainly used for debugging the MSP430 implementation. To execute the code, you must first generate a header file for the corresponding model. This header file is called `neural_network_parameters.h` and is created using the `convert_network.py` script in the directory above (without the `--msp` option). This header file must be copied into this directory. With the given parameters, you can compile and run the model.
```
$ make model
$ ./model <path-to-inputs> <path-to-labels>
```
The paths refer to the quantized txt files created by the script `create_mcu_dataset.py` in the directory above. Below is a concrete example of the execution command.
```
$ ./model ../../../data/pen_digits/folds_8/test_9_inputs.txt ../../../data/pen_digits/folds_8/test_9_labels.txt
```
