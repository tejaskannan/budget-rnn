# Layers
This module implements standard neural network layers.

## Neural Network Operations
The file `dense.py` implements dense neural network layers. The function `dense()` applies a single dense layer. The function `mlp()` uses dense to create multi-layer networks. These layers will optionally apply dropout regularization.

The `conv.py` file provides an implementation for a 1d convolution. The RNNs considered in the paper do not use this operation.

The `output_layers.py` file computes statistics for classification outputs. This process does not involve trainable parameters.

## Recurrent Neural Network (RNN) Cells
The `cells` directory contains implementations of the employed RNN cells. The file `cells/cell_utils.py` contains a [UGRNN](https://arxiv.org/pdf/1902.02390.pdf) cell. The files `cells/standard_rnn_cells.py`, `cells/budget_rnn_cells.py`, `cells/skip_rnn_cells.py`, and `cells/phased_rnn_cells.py` use the UGRNN implementation to create their respective cells.
