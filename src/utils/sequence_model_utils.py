from enum import Enum, auto
from typing import Optional


class SequenceModelType(Enum):
    RNN = auto()
    SKIP_RNN = auto()
    NBOW = auto()
    SAMPLE_RNN = auto()
    SAMPLE_NBOW = auto()


def is_sample(model_type: SequenceModelType) -> bool:
    return model_type in (SequenceModelType.SAMPLE_RNN, SequenceModelType.SAMPLE_NBOW)


def is_rnn(model_type: SequenceModelType) -> bool:
    return model_type in (SequenceModelType.RNN, SequenceModelType.SKIP_RNN, SequenceModelType.SAMPLE_RNN)


def is_nbow(model_type: SequenceModelType) -> bool:
    return model_type in (SequenceModelType.NBOW, SequenceModelType.SAMPLE_NBOW)
