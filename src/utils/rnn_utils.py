from enum import Enum, auto
from typing import Optional

from utils.constants import LOGITS, ACCURACY, PREDICTION, F1_SCORE , LOSS


RNN_CELL_NAME = 'rnn-cell'
RNN_LEVEL_NAME = 'rnn-level'
INPUT_NAME = 'input'
OUTPUT_LAYER_NAME = 'output-level'
GATES_NAME = 'gates'
STATES_NAME = 'states'
ALL_PREDICTIONS_NAME = 'predictions'
PREDICTION_PROB_NAME = 'prediction_probs'
EMBEDDING_NAME = 'embedding'
COMBINE_STATES_NAME = 'combine-states'
WHILE_LOOP_NAME = 'while-loop'


class RNNModelType(Enum):
    VANILLA = auto()
    SAMPLE = auto()
    CASCADE = auto()
    LINKED = auto()


def get_cell_level_name(level_index: int, should_share_weights: bool) -> str:
    if should_share_weights:
        return RNN_CELL_NAME
    return f'{RNN_CELL_NAME}-level-{level_index}'


def get_rnn_level_name(level_index: int) -> str:
    return f'{RNN_LEVEL_NAME}-{level_index}'


def get_input_name(level_index: int) -> str:
    return f'{INPUT_NAME}_{level_index}'


def get_output_layer_name(level_index: int) -> str:
    return f'{OUTPUT_LAYER_NAME}_{level_index}'


def get_loss_name(level_index: int) -> str:
    return f'{LOSS}_{level_index}'


def get_prediction_name(level_index: int) -> str:
    return f'{PREDICTION}_{level_index}'


def get_logits_name(level_index: int) -> str:
    return f'{LOGITS}_{level_index}'


def get_accuracy_name(level_index: int) -> str:
    return f'{ACCURACY}_{level_index}'


def get_prediction_prob_name(level_index: int) -> str:
    return f'{PREDICTION_PROB_NAME}_{level_index}'


def get_gates_name(level_index: int) -> str:
    return f'{GATES_NAME}_{level_index}'


def get_states_name(level_index: int) -> str:
    return f'{STATES_NAME}_{level_index}'


def get_f1_score_name(level_index: int) -> str:
    return f'{F1_SCORE}_{level_index}'


def get_embedding_name(level_index: int) -> str:
    return f'{EMBEDDING_NAME}-{level_index}'

def get_combine_states_name(name_prefix: Optional[str]) -> str:
    if name_prefix is None:
        return COMBINE_STATES_NAME
    return f'{name_prefix}-{COMBINE_STATES_NAME}'

def get_rnn_while_loop_name(name_prefix: Optional[str]) -> str:
    if name_prefix is None:
        return f'rnn-{WHILE_LOOP_NAME}'
    return f'{name_prefix}-{WHILE_LOOP_NAME}'
