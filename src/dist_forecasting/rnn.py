import numpy as np
from typing import Dict, Tuple
from utils.np_utils import leaky_relu, softmax, sigmoid


VAR_FMT = 'model/{0}:0'


def ugrnn_cell(input_features: np.ndarray, state: np.ndarray, model_vars: Dict[str, np.ndarray]) -> np.ndarray:
    W_embedding = model_vars[VAR_FMT.format('embedding-layer-kernel')]
    b_embedding = model_vars[VAR_FMT.format('embedding-layer-bias')]
    W_transform = model_vars[VAR_FMT.format('rnn_cell-W-transform')]
    b_transform = model_vars[VAR_FMT.format('rnn_cell-b-transform')]
    
    # Compute the input embedding, [B, D] array
    embedding = leaky_relu(np.matmul(input_features, W_embedding) + b_embedding)
    state_size = embedding.shape[-1]

    concat = np.concatenate([state, embedding], axis=-1)  # [B, 2 * D]
    transformed = np.matmul(concat, W_transform) + b_transform  # [B, 2 * D]

    # Pair of [B, D] arrays
    update, candidate = np.split(transformed, indices_or_sections=2, axis=-1)

    # Apply nonlinear functions
    update_gate = sigmoid(update + 1)
    candidate_state = np.tanh(candidate)

    return update_gate * state + (1.0 - update_gate) * candidate_state


def output_layer(states: np.ndarray, model_vars: Dict[str, np.ndarray]) -> np.ndarray:
    W_hidden = model_vars[VAR_FMT.format('output-layer-hidden-0-kernel')]
    b_hidden = model_vars[VAR_FMT.format('output-layer-hidden-0-bias')]
    W_output = model_vars[VAR_FMT.format('output-layer-output-kernel')]
    b_output = model_vars[VAR_FMT.format('output-layer-output-bias')]
 
    hidden = leaky_relu(np.matmul(states, W_hidden) + b_hidden)  # [B, T, K]
    pred = np.matmul(hidden, W_output) + b_output  # [B, T, C]
    return softmax(pred, axis=-1)


def rnn(inputs: np.ndarray, skip_mask: np.ndarray, model_vars: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    assert len(inputs.shape) == 3, 'Must provide 3D inputs'

    batch_size, seq_length = inputs.shape[0], inputs.shape[1]
    state_size = model_vars[VAR_FMT.format('embedding-layer-kernel')].shape[1]

    state = np.zeros(shape=(batch_size, state_size))
    states_list: List[np.ndarray] = []

    for seq_idx in range(seq_length):
        skip = skip_mask[seq_idx]
        next_state = ugrnn_cell(input_features=inputs[:, seq_idx, :],
                                state=state,
                                model_vars=model_vars)

        state = skip * next_state + (1.0 - skip) * state
        states_list.append(np.expand_dims(state, axis=1))

    states = np.concatenate(states_list, axis=1)  # [B, T, D]
    pred = output_layer(states=states, model_vars=model_vars)
    return pred, states
