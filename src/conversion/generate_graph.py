from argparse import ArgumentParser
from utils.file_utils import read_by_file_suffix
from typing import Dict, Any, List

from utils.constants import TRANSFORM_SEED, EMBEDDING_SEED, AGGREGATE_SEED, OUTPUT_SEED, UPDATE_SEED, RESET_SEED, CANDIDATE_SEED, FUSION_SEED


INCLUDE_HEADER = '#include "neural_network.h"\n'
METHOD_DEF = 'int8_t *execute_model(matrix *inputs[SEQ_LENGTH], int8_t *outputs) {\n'


def allocate_matrix(mat_name: str, num_rows: int, num_cols: int, prefix: str) -> str:
    lines: List[str] = []
    lines.append('{0}matrix {1}Mat;'.format(prefix, mat_name))
    lines.append('{0}dtype {1}Data[{2}];'.format(prefix, mat_name, num_rows * num_cols))
    lines.append('{0}{1}Mat.data = {1}Data;'.format(prefix, mat_name))
    lines.append('{0}{1}Mat.numRows = {2};'.format(prefix, mat_name, num_rows))
    lines.append('{0}{1}Mat.numCols = {2};'.format(prefix, mat_name, num_cols))
    lines.append('{0}matrix *{1} = &{1}Mat;'.format(prefix, mat_name)) 
    
    return '\n'.join(lines)


def create_tf_gru_cell(rnn_layer: Dict[str, str]) -> str:
    return '\tTFGRU rnn_cell = {{ {0}, {1}, {2}, {3} }};\n'.format(rnn_layer['gates_kernel'], rnn_layer['gates_bias'], rnn_layer['candidates_kernel'], rnn_layer['candidates_bias'])


def create_gru_cell(rnn_layer: Dict[str, str]) -> str:
    """
    Creates a GRU cell using the split-variable representation.
    """
    # Extract variables
    W_update, U_update, b_update = rnn_layer['W_update'], rnn_layer['U_update'], rnn_layer['b_update']
    W_reset, U_reset, b_reset = rnn_layer['W_reset'], rnn_layer['U_reset'], rnn_layer['b_reset']
    W_candidate, U_candidate, b_candidate = rnn_layer['W_candidate'], rnn_layer['U_candidate'], rnn_layer['b_candidate']

    return '\tGRU rnn_cell = {{ {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8} }};\n\n'.format(W_update,
                                                                                          U_update,
                                                                                          b_update,
                                                                                          W_reset,
                                                                                          U_reset,
                                                                                          b_reset,
                                                                                          W_candidate,
                                                                                          U_candidate,
                                                                                          b_candidate)


def create_fusion_layer(current_state: str, prev_state: str, temp_variable: str, fusion_layer: Dict[str, str]) -> str:
    """
    Layer to fuse together hidden states.
    """
    return current_state


def create_dense_layer(input_var: str, output_var: str, layer_info: List[Dict[str, str]], hidden_vars: List[str], seed: str, prefix: str, linear_output: bool) -> str:
    """
    Creates a dense layer using the C implementation.

    Args:
        output_var: Name of the result variable.
        layer_info: List of the variables at each layer
        prefix: Prefix for formatting purposes
    Returns:
        C code which executes this dense layer.
    """
    code: List[str] = []

    for i in range(len(layer_info)):
        W = layer_info[i]['kernel']
        bias = layer_info[i].get('bias', 'NULL_PTR')
        activation = layer_info[i].get('activation')

        # Create the activation function pointer
        if activation is None or (i == len(layer_info) - 1 and linear_output):
            activation = '&fp_linear'
        else:
            activation = '&fp_' + activation

        layer_var = output_var if i == len(layer_info) - 1 else hidden_vars[i]
        layer_seed = seed if i == len(layer_info) - 1 else '{0}{1}'.format(seed, i)

        code.append('{0}{1} = dense({1}, {2}, {3}, {4}, {5}, {6}, "{7}", {8});'.format(prefix, layer_var, input_var, W, bias, activation, 'IS_COMPRESSED', layer_seed, 'FIXED_POINT_PRECISION'))
        input_var = layer_var

    return '\n'.join(code)


def write_standard_graph(model_params: Dict[str, Any]):
    state_size = model_params['state_size']

    with open('neural_network.c', 'w') as output_file:
        output_file.write(INCLUDE_HEADER)

        # Create function definition and enclosing for loop
        output_file.write(METHOD_DEF)

        output_file.write(create_tf_gru_cell(model_params['transform']))
        output_file.write('\n')

        # Allocate temporary variables
        temp_var_names = ['transformed', 'state', 'temp_state']
        for temp_var in temp_var_names:
            output_file.write('{0}\n\n'.format(allocate_matrix(temp_var, state_size, 1, prefix='\t')))

        output_file.write('\tTFGRUTempStates gruTemp;\n')
        temp_vars = ['stacked', 'gates', 'candidate', 'update', 'reset', 'tempGate']
        for temp_var in temp_vars:
            size = 2 * state_size if temp_var in {'stacked', 'gates'} else state_size
            output_file.write('{0}\n'.format(allocate_matrix('gruTemp{0}'.format(temp_var.capitalize()), size, 1, prefix='\t')))
            output_file.write('\tgruTemp.{0} = gruTemp{1};\n\n'.format(temp_var, temp_var.capitalize()))

        output_hidden_units = model_params['output_hidden_units']
        output_hidden_vars: List[str] = []
        for i in range(len(output_hidden_units)):
            name = 'outputTemp{0}'.format(i)
            output_hidden_vars.append(name)
            output_file.write('{0}\n\n'.format(allocate_matrix(name, output_hidden_units[i], 1, prefix='\t')))

        num_outputs = model_params['output_units']
        output_file.write('{0}\n\n'.format(allocate_matrix('output', num_outputs, 1, prefix='\t')))

        # Initialize state to all zeros
        output_file.write('\tmatrix_set(state, 0);\n')

        output_file.write('\tuint16_t i;\n')
        output_file.write('\tfor (i = 0; i < SEQ_LENGTH; i++) {\n')
        output_file.write('\t\tmatrix *input = inputs[i];\n')

        # Create the embedding layer. We don't support hidden layers (yet)
        embedding_layer = create_dense_layer('input', 'transformed', model_params['embedding'], hidden_vars=[], seed=EMBEDDING_SEED, prefix='\t\t', linear_output=False)
        output_file.write(embedding_layer)
        output_file.write('\n')

        # Create the transformation layer
        output_file.write('\t\ttemp_state = apply_tf_gru(temp_state, transformed, state, &rnn_cell, &gruTemp, FIXED_POINT_PRECISION);\n')
        output_file.write('\t\tstate = matrix_replace(state, temp_state);\n')
        output_file.write('\t}\n')

        # Create the output layer
        output_layer = create_dense_layer('state', 'output', model_params['output'], hidden_vars=output_hidden_vars, seed=OUTPUT_SEED, prefix='\t', linear_output=True)
        output_file.write(output_layer)
        output_file.write('\n\n')

        if model_params['output_type'] == 'multi_classification':
            output_file.write('\tint16_t prediction = argmax(output);\n')
        elif model_params['output_type'] == 'binary_classification':
            output_file.write('\tint16_t prediction = (int8_t) output->data[0] > 0;\n')
        else:
            output_file.write('\tint16_t prediction = output->data[0];\n')

        # Save in output array
        output_file.write('\t*outputs = prediction;\n')

        ## Free all intermediate states
        #output_file.write('\tmatrix_free({0});\n'.format('transformed'))
        #output_file.write('\tmatrix_free({0});\n'.format('state'))
        #output_file.write('\tmatrix_free({0});\n'.format('temp_state'))
        #output_file.write('\tmatrix_free({0});\n\n'.format('output'))

        output_file.write('\treturn outputs;\n')
        output_file.write('}')


def write_adaptive_graph(model_params: Dict[str, str]):
    seq_length = model_params['seq_length']
    samples_per_seq = model_params['samples_per_seq']
    num_sequences = int(seq_length / samples_per_seq)
    state_size = model_params['state_size']

    with open('neural_network.c', 'w') as output_file:
        output_file.write('#include "neural_network.h"\n')

        # Create function definition and enclosing for loop
        output_file.write('int8_t *execute_model(matrix *inputs[SEQ_LENGTH], int8_t *outputs) {\n')

        # Create the GRU Cell
        output_file.write(create_gru_cell(model_params['transform']))
        output_file.write('\n')

        # Allocate temporary variables
        temp_vars = ['transformed','state', 'temp_state', 'fusion_gate', 'gateTemp']
        for temp_var in temp_vars:
            output_file.write('{0}\n\n'.format(allocate_matrix(temp_var, state_size, 1, prefix='\t')))

        output_file.write('{0}\n\n'.format(allocate_matrix('fusion_stack', state_size * 2, 1, prefix='\t')))

        # Create output dense layers
        output_hidden_units = model_params['output_hidden_units']
        output_hidden_vars: List[str] = []
        for i in range(len(output_hidden_units)):
            name = 'outputTemp{0}'.format(i)
            output_hidden_vars.append(name)
            output_file.write('{0}\n\n'.format(allocate_matrix(name, output_hidden_units[i], 1, prefix='\t')))

        num_outputs = model_params['output_units']
        output_file.write('{0}\n\n'.format(allocate_matrix('output', num_outputs, 1, prefix='\t')))

        output_file.write('\tmatrix *prev_states[SAMPLES_PER_SEQ];\n')
        for i in range(samples_per_seq):
            output_file.write('{0}\n'.format(allocate_matrix('prevStates{0}'.format(i), state_size, 1, prefix='\t')))
            output_file.write('\tprev_states[{0}] = prevStates{0};\n\n'.format(i))

        # Create the GRU temporary state
        output_file.write('\tGRUTempStates gruTemp;\n')
    
        temp_vars = ['update', 'reset', 'candidate', 'inputTemp', 'gateTemp']
        for temp_var in temp_vars:
            output_file.write('{0}\n'.format(allocate_matrix('gruTemp{0}'.format(temp_var.capitalize()), state_size, 1, prefix='\t')))
            output_file.write('\tgruTemp.{0} = gruTemp{1};\n\n'.format(temp_var, temp_var.capitalize()))

        output_file.write('\tint16_t i, j;\n')
        output_file.write('\tfor (i = 0; i < NUM_SEQUENCES; i++) {\n')

        # Initialize state to zero at the beginning of each sequence
        output_file.write('\t\tmatrix_set(state, 0);\n')

        output_file.write('\t\tfor (j = 0; j < {0}; j++) {{\n'.format('SAMPLES_PER_SEQ'))

        # Fetch the input
        output_file.write('\t\t\tmatrix *input = inputs[j * {0} + i];\n'.format(num_sequences))

        # Apply the embedding layer
        embedding_layer = create_dense_layer('input', 'transformed', model_params['embedding'], hidden_vars=[], seed=EMBEDDING_SEED, prefix='\t\t\t', linear_output=False)
        output_file.write(embedding_layer)
        output_file.write('\n')

        # Apply the fusion gate
        output_file.write('\t\t\tif (i > 0) {\n')
        output_file.write('\t\t\t\tfusion_stack = stack(fusion_stack, state, prev_states[j]);\n')

        # For now, we only support single-layer RNN cells
        fusion_layer = create_dense_layer('fusion_stack', 'fusion_gate', model_params['fusion'], hidden_vars=[], seed='{0}0'.format(FUSION_SEED), prefix='\t\t\t\t', linear_output=False)
        output_file.write(fusion_layer)
        output_file.write('\n')

        output_file.write('\t\t\t\ttemp_state = apply_gate(temp_state, fusion_gate, state, prev_states[j], gateTemp, FIXED_POINT_PRECISION);\n')
        output_file.write('\t\t\t\tstate = matrix_replace(state, temp_state);\n')

        output_file.write('\t\t\t}\n')

        # (2) Create the transformation layer. We currently only support single layer RNN cells.
        output_file.write('\t\t\ttemp_state = apply_gru(temp_state, transformed, state, &rnn_cell, &gruTemp, IS_COMPRESSED, 0, FIXED_POINT_PRECISION);\n')
        output_file.write('\t\t\tstate = matrix_replace(state, temp_state);\n')
        output_file.write('\t\t\tmatrix_replace(prev_states[j], state);\n')
        output_file.write('\t\t}\n\n')

        # (3) Create the output layer
        output_layer = create_dense_layer('state', 'output', model_params['output'], hidden_vars=output_hidden_vars, seed=OUTPUT_SEED, prefix='\t\t', linear_output=True)
        output_file.write(output_layer)
        output_file.write('\n\n')

        if model_params['output_type'] == 'multi_classification':
            output_file.write('\t\tint16_t prediction = argmax(output);\n')
        elif model_params['output_type'] == 'binary_classification':
            output_file.write('\t\tint16_t prediction = (int16_t) output->data[0] > 0;\n')
        else:
            output_file.write('\t\tint16_t prediction = output->data[0];\n')

        output_file.write('\t\toutputs[i] = prediction;\n')

        output_file.write('\t}\n\n')

        output_file.write('\treturn outputs;\n')
        output_file.write('}\n')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model-file', type=str, required=True)
    args = parser.parse_args()

    model_params = read_by_file_suffix(args.model_file)

    if model_params['model_class'] == 'standard':
        write_standard_graph(model_params)
    else:
        write_adaptive_graph(model_params)
