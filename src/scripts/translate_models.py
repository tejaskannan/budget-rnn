"""
Script to translate the old-style models into those
with new cell types. The new implementation is more efficient
and does not change the model behavior.
"""
import os.path
import numpy as np
from argparse import ArgumentParser
from shutil import copy
from typing import Dict, Tuple

from utils.file_utils import extract_model_name, read_by_file_suffix, make_dir, save_by_file_suffix, iterate_files
from utils.constants import EMBEDDING_NAME, OUTPUT_LAYER_NAME, STOP_PREDICTION, RNN_CELL_NAME, MODEL_PATH, HYPERS_PATH, TRANSFORM_NAME
from utils.hyperparameters import HyperParameters


NAME_FORMAT = 'model/{0}-{1}:0'
KERNEL = 'kernel'
BIAS = 'bias'


def translate_hyperparameters(hypers_file: str) -> HyperParameters:
    hypers = HyperParameters.create_from_file(hypers_file)

    if hypers.model_params['model_type'] == 'cascade_rnn':
        hypers.model_params['stride_length'] = 1
        hypers.model_params['num_outputs'] = int(1.0 / hypers.model_params['sample_frac'])
        hypers.model_params['model_type'] = 'sample_rnn'
    elif hypers.model_params['model_type'] == 'sample_rnn':
        hypers.model_params['stride_length'] = int(1.0 / hypers.model_params['sample_frac'])
        hypers.model_params['num_outputs'] = int(1.0 / hypers.model_params['sample_frac'])

    return hypers


def translate_ugrnn(W_update: np.ndarray,
                    U_update: np.ndarray,
                    b_update: np.ndarray,
                    W_candidate: np.ndarray,
                    U_candidate: np.ndarray,
                    b_candidate: np.ndarray) -> Tuple[np.ndarray]:
    """
    Merges the trainable parameters of UGRNN cells. The W-transform matrix should have the following block structure:
        |W_update, W_candidate|
        |U_update, U_candidate|
    """
    state_mat = np.hstack([W_update, W_candidate])  # [D, 2*D]
    input_mat = np.hstack([U_update, U_candidate])  # [D, 2*D]
    W = np.vstack([state_mat, input_mat])  # [2 * D, 2 * D]

    b = np.vstack([b_update, b_candidate]).reshape(1, -1)  # [1, 2 * D]
    return W, b


def translate_adaptive_model(model_file: str) -> Dict[str, np.ndarray]:
    """
    Translates the model parameters to the new format. This is a mostly manual process.
    """
    # Fetch previous parameters
    model_parameters = read_by_file_suffix(model_file)

    # Dictionary to hold the parameters with translated names
    result: Dict[str, np.ndarray] = dict()
    
    # Embedding Layer
    result[NAME_FORMAT.format(EMBEDDING_NAME, KERNEL)] = model_parameters[NAME_FORMAT.format('embedding', KERNEL)]
    result[NAME_FORMAT.format(EMBEDDING_NAME, BIAS)] = model_parameters[NAME_FORMAT.format('embedding', BIAS)]

    # Output Layer
    new_hidden_name = '{0}-hidden-0'.format(OUTPUT_LAYER_NAME)
    old_hidden_name = 'output-level-hidden-0'
    result[NAME_FORMAT.format(new_hidden_name, KERNEL)] = model_parameters[NAME_FORMAT.format(old_hidden_name, KERNEL)]
    result[NAME_FORMAT.format(new_hidden_name, BIAS)] = model_parameters[NAME_FORMAT.format(old_hidden_name, BIAS)]

    new_output_name = '{0}-output'.format(OUTPUT_LAYER_NAME)
    old_output_name = 'output-level-output'
    result[NAME_FORMAT.format(new_output_name, KERNEL)] = model_parameters[NAME_FORMAT.format(old_output_name, KERNEL)]
    result[NAME_FORMAT.format(new_output_name, BIAS)] = model_parameters[NAME_FORMAT.format(old_output_name, BIAS)]

    # Copy the stop prediction layer (this has not changed)
    stop_hidden_name = '{0}-hidden-0'.format(STOP_PREDICTION)
    result[NAME_FORMAT.format(stop_hidden_name, KERNEL)] = model_parameters[NAME_FORMAT.format(stop_hidden_name, KERNEL)]
    result[NAME_FORMAT.format(stop_hidden_name, BIAS)] = model_parameters[NAME_FORMAT.format(stop_hidden_name, BIAS)]

    stop_output_name = '{0}-output'.format(STOP_PREDICTION)
    result[NAME_FORMAT.format(stop_output_name, KERNEL)] = model_parameters[NAME_FORMAT.format(stop_output_name, KERNEL)]
    result[NAME_FORMAT.format(stop_output_name, BIAS)] = model_parameters[NAME_FORMAT.format(stop_output_name, BIAS)]

    # Translate the RNN cell. This is a bit more intensive because of merged matrices in the new cells. For now, this
    # only works for UGRNN cells
    W_update = model_parameters[NAME_FORMAT.format('rnn-cell-cell-0-W-update', KERNEL)]
    U_update = model_parameters[NAME_FORMAT.format('rnn-cell-cell-0-U-update', KERNEL)]
    b_update = model_parameters[NAME_FORMAT.format('rnn-cell-cell-0-b-update', BIAS)]

    W_candidate = model_parameters[NAME_FORMAT.format('rnn-cell-cell-0-W', KERNEL)]
    U_candidate = model_parameters[NAME_FORMAT.format('rnn-cell-cell-0-U', KERNEL)]
    b_candidate = model_parameters[NAME_FORMAT.format('rnn-cell-cell-0-b', BIAS)]
    
    new_W_name = 'model/{0}-W-transform:0'.format(RNN_CELL_NAME)
    new_b_name = 'model/{0}-b-transform:0'.format(RNN_CELL_NAME)

    W, b = translate_ugrnn(W_update=W_update,
                           U_update=U_update,
                           b_update=b_update,
                           W_candidate=W_candidate,
                           U_candidate=U_candidate,
                           b_candidate=b_candidate)
    result[new_W_name] = W
    result[new_b_name] = b

    # The fusion layer is a part of the cell in the new version
    combine_kernel = NAME_FORMAT.format('combine-states-0', KERNEL)
    if combine_kernel in model_parameters:
        W_fusion = model_parameters[combine_kernel]
        b_fusion = model_parameters[NAME_FORMAT.format('combine-states-0', BIAS)]

        new_W_fusion_name = 'model/{0}-W-fusion:0'.format(RNN_CELL_NAME)
        new_b_fusion_name = 'model/{0}-b-fusion:0'.format(RNN_CELL_NAME)
        result[new_W_fusion_name] = W_fusion
        result[new_b_fusion_name] = b_fusion

    return result


def translate_skip_or_phased_model(model_file: str) -> Dict[str, np.ndarray]:
    model_parameters = read_by_file_suffix(model_file)

    # Copy all non-RNN cell weight matrices
    result: Dict[str, np.ndarray] = dict()
    for var_name, var in model_parameters.items():
        if TRANSFORM_NAME not in var_name:
            result[var_name] = var

    # The RNN cell now stacks the transformation layers together. We perform this
    # translation with the given variables. This only works for UGRNN Cells (currently)
    W_update = model_parameters[NAME_FORMAT.format('transform-layer-W-update', KERNEL)]
    U_update = model_parameters[NAME_FORMAT.format('transform-layer-U-update', KERNEL)]
    b_update = model_parameters[NAME_FORMAT.format('transform-layer-b-update', BIAS)]

    W_candidate = model_parameters[NAME_FORMAT.format('transform-layer-W', KERNEL)]
    U_candidate = model_parameters[NAME_FORMAT.format('transform-layer-U', KERNEL)]
    b_candidate = model_parameters[NAME_FORMAT.format('transform-layer-b', BIAS)]

    W, b = translate_ugrnn(W_update=W_update,
                           U_update=U_update,
                           b_update=b_update,
                           W_candidate=W_candidate,
                           U_candidate=U_candidate,
                           b_candidate=b_candidate)

    new_W_name = 'model/{0}-W-transform:0'.format(RNN_CELL_NAME)
    new_b_name = 'model/{0}-b-transform:0'.format(RNN_CELL_NAME)
    result[new_W_name] = W
    result[new_b_name] = b

    # Translate the state layer (if needed)
    W_state_name = NAME_FORMAT.format('transform-layer-W-state', KERNEL)
    if W_state_name in model_parameters:
        new_W_state_name = 'model/{0}-W-state:0'.format(RNN_CELL_NAME)
        result[new_W_state_name] = model_parameters[W_state_name]

        b_state_name = NAME_FORMAT.format('transform-layer-b-state', BIAS)
        new_b_state_name = 'model/{0}-b-state:0'.format(RNN_CELL_NAME)
        result[new_b_state_name] = model_parameters[b_state_name]

    return result


def translate_standard_model(model_file: str) -> Dict[str, np.ndarray]:
    """
    Translates a Standard RNN model to the new format. This mainly
    has to do with renaming variables.

    Args:
        model_file: Path to the model parameters file
    Returns:
        A translated dictionary of model parameters.
    """
    model_parameters = read_by_file_suffix(model_file)

    # Copy all non-RNN cell weight matrices
    result: Dict[str, np.ndarray] = dict()
    for var_name, var in model_parameters.items():
        if 'rnn' not in var_name:
            result[var_name] = var

    # Translate the RNN cell. For now, this only works for UGRNN cells.
    new_W_name = 'model/{0}-W-transform:0'.format(RNN_CELL_NAME)
    old_W_name = 'model/rnn/multi_rnn_cell/cell_0/ugrnn_cell/kernel:0'

    old_W = model_parameters[old_W_name]
    W_input, W_state = np.split(old_W, indices_or_sections=2, axis=0)  # Pair of [D, 2*D] tensors
    new_W = np.vstack([W_state, W_input])  # [2*D, 2*D]
    result[new_W_name] = new_W

    print(new_W.shape)

    new_b_name = 'model/{0}-b-transform:0'.format(RNN_CELL_NAME)
    old_b_name = 'model/rnn/multi_rnn_cell/cell_0/ugrnn_cell/bias:0'
    result[new_b_name] = np.expand_dims(model_parameters[old_b_name], axis=0)  # [1, 2*D]

    return result


def translate_model(model_file: str, output_folder: str):
    make_dir(output_folder)
    
    # Extract the model name
    save_folder, model_file_name = os.path.split(model_file)
    model_name = extract_model_name(model_file_name)

    hypers_file = os.path.join(save_folder, HYPERS_PATH.format(model_name))
    translated_hypers = translate_hyperparameters(hypers_file)
    save_by_file_suffix(translated_hypers.__dict__(), os.path.join(output_folder, HYPERS_PATH.format(model_name)))

    # Translate and save the model parameters
    model_type = translated_hypers.model_params['model_type']
    if model_type in ('skip_rnn', 'phased_rnn'):
        translated_params = translate_skip_or_phased_model(model_file)
    elif model_type == 'rnn':
        translated_params = translate_standard_model(model_file)
    elif model_type == 'sample_rnn':
        translated_params = translate_adaptive_model(model_file)
    else:
        raise ValueError('Unknown model type: {0}'.format(model_type))

    save_by_file_suffix(translated_params, os.path.join(output_folder, MODEL_PATH.format(model_name)))

    # Copy all other files
    for file_path in iterate_files(save_folder, 'model.*{0}'.format(model_name)):
        if file_path not in (model_file, hypers_file):
            _, file_name = os.path.split(file_path)
            copy(file_path, os.path.join(output_folder, file_name))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    args = parser.parse_args()

    if os.path.isdir(args.model_path):
        model_paths = list(iterate_files(args.model_path, pattern='model-[A-Z]+.*_model_best\.pkl\.gz'))
    else:
        model_paths = [args.model_path]

    for model_file in model_paths:
        translate_model(model_file, args.output_folder)
