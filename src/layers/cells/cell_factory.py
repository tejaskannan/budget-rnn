import tensorflow as tf
from enum import Enum, auto
from typing import Dict, Any

from .skip_rnn_cells import SkipUGRNNCell, SkipGRUCell
from .sample_rnn_cells import SampleUGRNNCell
from .standard_rnn_cells import UGRNNCell
from .phased_rnn_cells import PhasedUGRNNCell
from utils.tfutils import get_activation


class CellClass(Enum):
    STANDARD = auto()
    SKIP = auto()
    SAMPLE = auto()
    PHASED = auto()


class CellType(Enum):
    GRU = auto()
    UGRNN = auto()


def make_rnn_cell(cell_class: CellClass,
                  cell_type: CellType,
                  units: int,
                  activation: str,
                  name: str,
                  recurrent_noise: tf.Tensor,
                  **kwargs: Dict[str, Any]) -> tf.nn.rnn_cell.RNNCell:
    """
    Creates an RNN Cell using the given parameters.
    """
    if cell_class == CellClass.STANDARD:
        if cell_type == CellType.UGRNN:
            return UGRNNCell(units=units, activation=activation, name=name, recurrent_noise=recurrent_noise)
    elif cell_class == CellClass.SKIP:
        if cell_type == CellType.UGRNN:
            return SkipUGRNNCell(units=units, activation=activation, name=name, recurrent_noise=recurrent_noise)
        elif cell_type == CellType.GRU:
            return SkipGRUCell(units=units, activation=activation, name=name, recurrent_noise=recurrent_noise)
    elif cell_class == CellClass.SAMPLE:
        if cell_type == CellType.UGRNN:
            return SampleUGRNNCell(units=units, activation=activation, name=name, recurrent_noise=recurrent_noise)
    elif cell_class == CellClass.PHASED:
        if cell_type == CellType.UGRNN:
            return PhasedUGRNNCell(units=units,
                                   activation=activation,
                                   recurrent_noise=recurrent_noise,
                                   on_fraction=kwargs['on_fraction'],
                                   leak_rate=kwargs['leak_rate'],
                                   period_init=kwargs['period_init'],
                                   name=name)

    raise ValueError('Unknown cell class {0} and type {1}'.format(cell_class.name, cell_type.name))
