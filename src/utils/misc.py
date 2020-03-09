import numpy as np
from dpu_utils.mlutils import Vocabulary
from typing import List, Any, Optional, Dict


def get_token_for_id_multiple(vocab: Vocabulary, ids: List[Any], extended_vocab: Optional[Dict[int, str]] = None) -> List[Any]:
    if extended_vocab is None:
        return [vocab.get_name_for_id(i) for i in ids]

    tokens: List[str] = []
    for token_id in ids:
        if token_id in extended_vocab:
            tokens.append(extended_vocab[token_id])
        elif token_id < len(vocab):
            tokens.append(vocab.get_name_for_id(token_id))
        else:
            tokens.append(vocab.get_unk())
    return tokens


def pad_array(arr: np.array, new_size: int, value: Any, axis: int) -> np.array:
    pad_width = new_size - arr.shape[axis]
    if pad_width <= 0 :
        return arr

    widths = [(0, 0) for _ in range(len(arr.shape))]
    widths[axis] = (0, pad_width)
    return np.pad(arr, widths, mode='constant', constant_values=value)

def softmax(arr: np.ndarray):
    exp_array = np.exp(arr)
    return exp_array / np.sum(exp_array)

def sigmoid(arr: np.ndarray):
    return 1.0 / (1.0 + np.exp(-arr))
