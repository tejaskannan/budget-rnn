"""
Script to translate the old-style models into those
with new cell types. The new implementation is more efficient
and does not change the model correctness.
"""
import os.path
import numpy as np
from argparse import ArgumentParser
from shutil import copy
from typing import Dict

from utils.file_utils import extract_model_name, read_by_file_suffix, make_dir, save_by_file_suffix, iterate_files
from utils.constants import EMBEDDING_NAME, OUTPUT_LAYER_NAME, STOP_PREDICTION, RNN_CELL_NAME, MODEL_PATH, HYPERS_PATH
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
    else:
        hypers.model_params['stride_length'] = int(1.0 / hypers.model_params['sample_frac'])
        hypers.model_params['num_outputs'] = int(1.0 / hypers.model_params['sample_frac'])

    return hypers


def translate_model_parameters(model_file: str) -> Dict[str, np.ndarray]:
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
    
    new_W_name = 'model/{0}-W-update:0'.format(RNN_CELL_NAME)
    new_b_name = 'model/{0}-b-update:0'.format(RNN_CELL_NAME)
    result[new_W_name] = np.vstack([W_update, U_update])  # [2*D, D]
    result[new_b_name] = b_update

    W_candidate = model_parameters[NAME_FORMAT.format('rnn-cell-cell-0-W', KERNEL)]
    U_candidate = model_parameters[NAME_FORMAT.format('rnn-cell-cell-0-U', KERNEL)]
    b_candidate = model_parameters[NAME_FORMAT.format('rnn-cell-cell-0-b', BIAS)]
    
    new_W_name = 'model/{0}-W-candidate:0'.format(RNN_CELL_NAME)
    new_b_name = 'model/{0}-b-candidate:0'.format(RNN_CELL_NAME)
    result[new_W_name] = np.vstack([W_candidate, U_candidate])  # [2*D, D]
    result[new_b_name] = b_candidate

    # The fusion layer is a part of the cell in the new version
    combine_kernel = NAME_FORMAT.format('combine-states-0', KERNEL)
    if combine_kernel in model_parameters:
        W_fusion = model_parameters[combine_kernel]
        b_fusion = model_parameters[NAME_FORMAT.format('combine-states-0', BIAS)]

        new_W_name = 'model/{0}-W-fusion:0'.format(RNN_CELL_NAME)
        new_b_name = 'model/{0}-b-fusion:0'.format(RNN_CELL_NAME)
        result[new_W_name] = W_fusion
        result[new_b_name] = b_fusion

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
    translated_params = translate_model_parameters(model_file)
    save_by_file_suffix(translated_params, os.path.join(output_folder, MODEL_PATH.format(model_name)))

    # Copy all other files
    for file_path in iterate_files(save_folder, 'model.*{0}'.format(model_name)):
        if file_path not in (model_file, hypers_file):
            _, file_name = os.path.split(file_path)
            copy(file_path, os.path.join(output_folder, file_name))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-file', type=str, required=True)
    parser.add_argument('--output-folder', type=str, required=True)
    args = parser.parse_args()

    translate_model(args.model_file, args.output_folder)
