# Models
This module implements the sequential neural network models.

## TF Model
The `base_model.py` and `tf_model.py` files implement standard operations for inference models. The `tf_model.py` file contains code for building (see `make()`) and training (see `train()`) models in Tensorflow. The `TFModel` class also manages the dataset-specific metadata (e.g. for data normalization).

## Standard Model
The `standard_model.py` file implements the computation graph for the non-Budget RNN sequential models. This qualifier includes Skip and Phased RNNs. The class also contains implementations for non-RNN models such as Neural Bag-of-Words (`NBOW`) and 1D `CNNs`. The paper does not focus on the non-RNN models.

## Adaptive Model
The `adaptiive_model.py` file contains code relevant to building and training Budget RNNs. This class manages information specific to Budget RNN. For example, it implements the loss function that accounts for halting signals (see `make_loss()`). Furthermore, the `_make_rnn_model()` method manages the subsequence creation based on the Budget RNN stride length.
