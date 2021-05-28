# Model Conversion

This module contains code to convert neural network parameters into variables stored in a C header file. There are two files in this directory, and these files have the following uses.

## Conversion Utils
The ```conversion_utils.py``` file contains utility functions for model conversion. These utilities include converting weight matrices to fixed point integers and writing static variables according to C syntax.

## Layer Conversion
The file ```layer_conversion.py``` contains a function to convert the weight matrix of a single neural network layer. This function handles details around matrix transposition needed when converting between the Tensorflow implementation and the C implementation.
