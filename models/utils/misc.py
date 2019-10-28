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
