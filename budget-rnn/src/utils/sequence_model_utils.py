from enum import Enum, auto
from typing import Optional


class SequenceModelType(Enum):
    RNN = auto()
    SKIP_RNN = auto()
    PHASED_RNN = auto()
    NBOW = auto()
    CONV = auto()
    BUDGET_RNN = auto()
    BUDGET_NBOW = auto()
    BUDGET_CONV = auto()


def is_budget(model_type: SequenceModelType) -> bool:
    return model_type in (SequenceModelType.BUDGET_RNN, SequenceModelType.BUDGET_NBOW, SequenceModelType.BUDGET_CONV)


def is_rnn(model_type: SequenceModelType) -> bool:
    return model_type in (SequenceModelType.RNN, SequenceModelType.SKIP_RNN, SequenceModelType.BUDGET_RNN, SequenceModelType.PHASED_RNN)


def is_nbow(model_type: SequenceModelType) -> bool:
    return model_type in (SequenceModelType.NBOW, SequenceModelType.BUDGET_NBOW)


def is_conv(model_type: SequenceModelType) -> bool:
    return model_type in (SequenceModelType.CONV, SequenceModelType.BUDGET_CONV)
