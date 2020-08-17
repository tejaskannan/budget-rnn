import tensorflow as tf
from enum import Enum, auto

from .skip_rnn_cells import SkipUGRNNCell
from .sample_rnn_cells import SampleUGRNNCell
from utils.tfutils import get_activation


class CellClass(Enum):
    STANDARD = auto()
    SKIP = auto()
    SAMPLE = auto()


class CellType(Enum):
    GRU = auto()
    UGRNN = auto()


def make_rnn_cell(cell_class: CellClass, cell_type: CellType, units: int, activation: str, name: str) -> tf.nn.rnn_cell.RNNCell:
    """
    Creates an RNN Cell using the given parameters.
    """
    if cell_class == CellClass.STANDARD:
        if cell_type == CellType.GRU:
            return tf.nn.rnn_cell.GRUCell(num_units=units,
                                          activation=get_activation(activation),
                                          name=name)
        elif cell_type == CellType.UGRNN:
            return tf.contrib.rnn.UGRNNCell(num_units=units,
                                            initializer=tf.glorot_uniform_initializer())
        else:
            raise ValueError('Unknown cell class {0} and type {1}'.format(cell_class.name, cell_type.name))
    elif cell_class == CellClass.SKIP:
        if cell_type == CellType.UGRNN:
            return SkipUGRNNCell(units=units, activation=activation, name=name)
        else:
            raise ValueError('Unknown cell class {0} and type {1}'.format(cell_class.name, cell_type.name))
    elif cell_class == CellClass.SAMPLE:
        if cell_type == CellType.UGRNN:
            return SampleUGRNNCell(units=units, activation=activation, name=name)
        else:
            raise ValueError('Unknown cell class {0} and type {1}'.format(cell_class.name, cell_type.name))

    raise ValueError('Unknown cell class {0}'.format(cell_class.name))
