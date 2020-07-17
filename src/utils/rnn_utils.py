from enum import Enum, auto
from typing import Optional

from utils.constants import LOGITS, ACCURACY, PREDICTION, F1_SCORE , LOSS


BACKWARDS_NAME = 'backward'
RNN_CELL_NAME = 'rnn-cell'
RNN_LEVEL_NAME = 'rnn-level'
INPUT_NAME = 'input'
OUTPUT_LAYER_NAME = 'output-level'
GATES_NAME = 'gates'
STATES_NAME = 'states'
ALL_PREDICTIONS_NAME = 'predictions'
PREDICTION_PROB_NAME = 'prediction_probs'
EMBEDDING_NAME = 'embedding'
TRANSFORM_NAME = 'transform'
AGGREGATION_NAME = 'aggregation'
COMBINE_STATES_NAME = 'combine-states'
WHILE_LOOP_NAME = 'while-loop'
STOP_OUTPUT_NAME = 'stop_output'
STOP_PREDICTION = 'stop-prediction'
OUTPUT_ATTENTION = 'output-attention'


class AdaptiveModelType(Enum):
    INDEPENDENT_RNN = auto()
    INDEPENDENT_NBOW = auto()
    INDEPENDENT_BIRNN = auto()
    SAMPLE_RNN = auto()
    CASCADE_RNN = auto()
    BIDIR_SAMPLE = auto()
    SAMPLE_NBOW = auto()
    CASCADE_NBOW = auto()


def is_cascade(model_type: AdaptiveModelType) -> bool:
    return model_type in (AdaptiveModelType.CASCADE_RNN, AdaptiveModelType.CASCADE_NBOW)


def is_sample(model_type: AdaptiveModelType) -> bool:
    return model_type in (AdaptiveModelType.SAMPLE_RNN, AdaptiveModelType.SAMPLE_NBOW)


def is_rnn(model_type: AdaptiveModelType) -> bool:
    return model_type in (AdaptiveModelType.INDEPENDENT_RNN, AdaptiveModelType.INDEPENDENT_BIRNN, AdaptiveModelType.SAMPLE_RNN, AdpativeModelType.CASCADE_RNN)


def is_nbow(model_type: AdaptiveModelType) -> bool:
    return model_type in (AdaptiveModelType.INDEPENDENT_NBOW, AdaptiveModelType.SAMPLE_NBOW, AdaptiveModelType.CASCADE_NBOW)


def is_independent(model_type: AdaptiveModelType) -> bool:
    return model_type in (AdaptiveModelType.INDEPENDENT_NBOW, AdaptiveModelType.INDEPENDENT_RNN, AdaptiveModelType.INDEPENDENT_BIRNN)


def is_bidirectional(model_type: AdaptiveModelType) -> bool:
    return model_type in (AdaptiveModelType.BIDIR_SAMPLE, AdaptiveModelType.INDEPENDENT_BIRNN)


def get_cell_level_name(level_index: int, should_share_weights: bool) -> str:
    if should_share_weights:
        return RNN_CELL_NAME
    return f'{RNN_CELL_NAME}-level-{level_index}'


def get_rnn_level_name(level_index: int) -> str:
    return f'{RNN_LEVEL_NAME}-{level_index}'


def get_backward_name(name: str) -> str:
    return '{0}-{1}'.format(name, BACKWARDS_NAME)


def get_input_name(level_index: int) -> str:
    return f'{INPUT_NAME}_{level_index}'


def get_output_layer_name(level_index: int, should_share_weights: bool) -> str:
    if should_share_weights:
        return OUTPUT_LAYER_NAME
    return f'{OUTPUT_LAYER_NAME}_{level_index}'


def get_stop_output_name(level_index: int) -> str:
    return '{0}_{1}'.format(STOP_OUTPUT_NAME, level_index)


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


def get_transform_name(level_index: int, should_share_weights: bool) -> str:
    if should_share_weights:
        return TRANSFORM_NAME
    return f'{TRANSFORM_NAME}_{level_index}'


def get_aggregation_name(level_index: int, should_share_weights: bool) -> str:
    if should_share_weights:
        return AGGREGATION_NAME
    return f'{AGGREGATION_NAME}_{level_index}'


def get_embedding_name(level_index: int, should_share_weights: bool) -> str:
    if should_share_weights:
        return EMBEDDING_NAME
    return '{0}_level_{1}'.format(EMBEDDING_NAME, level_index)


def get_combine_states_name(name_prefix: Optional[str], should_share_weights: bool) -> str:
    if name_prefix is None or should_share_weights:
        return COMBINE_STATES_NAME
    return f'{name_prefix}-{COMBINE_STATES_NAME}'


def get_rnn_while_loop_name(name_prefix: Optional[str]) -> str:
    if name_prefix is None:
        return f'rnn-{WHILE_LOOP_NAME}'
    return f'{name_prefix}-{WHILE_LOOP_NAME}'
