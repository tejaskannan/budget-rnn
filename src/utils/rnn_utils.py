from enum import Enum, auto


RNN_CELL_NAME = 'rnn-cell'
RNN_LEVEL_NAME = 'rnn-level'
INPUT_NAME = 'input'
OUTPUT_LAYER_NAME = 'output-level'
LOSS_NAME = 'loss'
GATES_NAME = 'gates'
STATES_NAME = 'states'
ALL_PREDICTIONS_NAME = 'predictions'
PREDICTION_NAME = 'prediction'
PREDICTION_PROB_NAME = 'prediction_probs'
LOGITS_NAME = 'logits'
ACCURACY_NAME = 'accuracy'


class RNNModelType(Enum):
    VANILLA = auto()
    SAMPLE = auto()
    CASCADE = auto()


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
    return f'{LOSS_NAME}_{level_index}'


def get_prediction_name(level_index: int) -> str:
    return f'{PREDICTION_NAME}_{level_index}'


def get_logits_name(level_index: int) -> str:
    return f'{LOGITS_NAME}_{level_index}'


def get_accuracy_name(level_index: int) -> str:
    return f'{ACCURACY_NAME}_{level_index}'


def get_prediction_prob_name(level_index: int) -> str:
    return f'{PREDICTION_PROB_NAME}_{level_index}'


def get_gates_name(level_index: int) -> str:
    return f'{GATES_NAME}_{level_index}'


def get_states_name(level_index: int) -> str:
    return f'{STATES_NAME}_{level_index}'
