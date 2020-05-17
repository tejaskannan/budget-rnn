from argparse import ArgumentParser
from utils.file_utils import read_by_file_suffix
from typing import Dict, Any, List


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

    return '\tGRU rnn_cell = {{ {0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8} }};\n'.format(W_update,
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


def create_dense_layer(input_var: str, output_var: str, layer_info: List[Dict[str, str]], prefix: str) -> str:
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
        if activation is None:
            activation = 'NULL_PTR'
        else:
            activation = '&fp_' + activation

        if i < len(layer_info) - 1:
            layer_var = 'temp{0}'.format(i)
            code.append('{0}matrix *{1} = matrix_allocate({2}->numRows, {3}->numCols);'.format(prefix, layer_var, W, input_var))
        else:
            layer_var = output_var

        code.append('{0}{1} = dense({1}, {2}, {3}, {4}, {5}, {6});'.format(prefix, layer_var, input_var, W, bias, activation, 'FIXED_POINT_PRECISION'))
        input_var = layer_var

    for i in range(len(layer_info) - 1):
        code.append('{0}matrix_free(temp{1});'.format(prefix, i))

    return '\n'.join(code)


def write_standard_graph(model_params: Dict[str, Any]):
    with open('neural_network.c', 'w') as output_file:
        output_file.write('#include "neural_network.h"\n')

        # Create function definition and enclosing for loop
        output_file.write('int16_t *execute_model(matrix *inputs[SEQ_LENGTH], int16_t *outputs) {\n')

        output_file.write(create_tf_gru_cell(model_params['transform']))
        output_file.write('\n')

        output_file.write('\tmatrix *transformed = matrix_allocate({0}, 1);\n'.format(model_params['state_size']))
        output_file.write('\tmatrix *state = matrix_allocate({0}, 1);\n'.format(model_params['state_size']))
        output_file.write('\tmatrix_set(state, 0);\n')
        output_file.write('\tmatrix *temp_state = matrix_allocate({0}, 1);\n\n'.format(model_params['state_size']))

        output_file.write('\tfor (int16_t i = 0; i < SEQ_LENGTH; i++) {\n')
        output_file.write('\t\tmatrix *input = inputs[i];\n')

        # (1) Create the embedding layer
        embedding_layer = create_dense_layer('input', 'transformed', model_params['embedding'], prefix='\t\t')
        output_file.write(embedding_layer)
        output_file.write('\n')

        # (2) Create the transformation layer
        output_file.write('\t\ttemp_state = apply_tf_gru(temp_state, transformed, state, &rnn_cell, FIXED_POINT_PRECISION);\n')
        output_file.write('\t\tstate = matrix_replace(state, temp_state);\n')
        output_file.write('\t}\n')

        # (3) Create the output layer
        output_kernel = model_params['output'][-1]['kernel']
        output_file.write('\n\tmatrix *output = matrix_allocate({0}->numRows, 1);\n'.format(output_kernel))
        output_layer = create_dense_layer('state', 'output', model_params['output'], prefix='\t')
        output_file.write(output_layer)
        output_file.write('\n\n')

        if model_params['output_type'] == 'multi_classification':
            output_file.write('\tint16_t prediction = argmax(output);\n')
        elif model_params['output_type'] == 'binary_classification':
            output_file.write('\tint16_t prediction = (int16_t) output->data[0] > 0;\n')
        else:
            output_file.write('\tint16_t prediction = output->data[0];\n')

        output_file.write('\t*outputs = prediction;\n')

        # Free all intermediate states
        output_file.write('\tmatrix_free({0});\n'.format('transformed'))
        output_file.write('\tmatrix_free({0});\n'.format('state'))
        output_file.write('\tmatrix_free({0});\n'.format('temp_state'))
        output_file.write('\tmatrix_free({0});\n\n'.format('output'))

        output_file.write('\treturn outputs;\n')
        output_file.write('}')


def write_adaptive_graph(model_params: Dict[str, str]):
    seq_length = model_params['seq_length']
    samples_per_seq = model_params['samples_per_seq']
    num_sequences = int(seq_length / samples_per_seq)
    state_size = model_params['state_size']

    print(num_sequences)
    with open('neural_network.c', 'w') as output_file:
        output_file.write('#include "neural_network.h"\n')

        # Create function definition and enclosing for loop
        output_file.write('int16_t *execute_model(matrix *inputs[SEQ_LENGTH], int16_t *outputs) {\n')

        # Create the GRU Cell
        output_file.write(create_gru_cell(model_params['transform']))
        output_file.write('\n')

        # Allocate temporary variables
        output_file.write('\tmatrix *transformed = matrix_allocate({0}, 1);\n'.format(state_size))
        output_file.write('\tmatrix *state = matrix_allocate({0}, 1);\n'.format(state_size))
        output_file.write('\tmatrix *temp_state = matrix_allocate({0}, 1);\n'.format(state_size))
        output_file.write('\tmatrix *fusion_stack = matrix_allocate(2 * {0}, 1);\n'.format(state_size))
        output_file.write('\tmatrix *fusion_gate = matrix_allocate({0}, 1);\n\n'.format(state_size))

        output_kernel = model_params['output'][-1]['kernel']
        output_file.write('\tmatrix *output = matrix_allocate({0}->numRows, 1);\n\n'.format(output_kernel))

        output_file.write('\tmatrix *prev_states[SAMPLES_PER_SEQ];\n')
        output_file.write('\tfor (int16_t i = 0; i < SAMPLES_PER_SEQ; i++) {\n')
        output_file.write('\t\tprev_states[i] = matrix_allocate({0}, 1);\n'.format(state_size))
        output_file.write('\t}\n\n')

        output_file.write('\tfor (int16_t i = 0; i < {0}; i++) {{\n'.format(num_sequences))

        # Initialize state to zero at the beginning of each sequence
        output_file.write('\t\tmatrix_set(state, 0);\n')

        output_file.write('\t\tfor (int16_t j = 0; j < {0}; j++) {{\n'.format('SAMPLES_PER_SEQ'))

        # Fetch the input
        output_file.write('\t\t\tmatrix *input = inputs[j * {0} + i];\n'.format(num_sequences))

        # Apply the embedding layer
        embedding_layer = create_dense_layer('input', 'transformed', model_params['embedding'], prefix='\t\t\t')
        output_file.write(embedding_layer)
        output_file.write('\n')

        # Apply the fusion gate
        output_file.write('\t\t\tif (i > 0) {\n')
        output_file.write('\t\t\t\tfusion_stack = stack(fusion_stack, state, prev_states[j]);\n')

        fusion_layer = create_dense_layer('fusion_stack', 'fusion_gate', model_params['fusion'], prefix='\t\t\t\t')
        output_file.write(fusion_layer)
        output_file.write('\n')

        output_file.write('\t\t\t\ttemp_state = apply_gate(temp_state, fusion_gate, state, prev_states[j], FIXED_POINT_PRECISION);\n')
        output_file.write('\t\t\t\tstate = matrix_replace(state, temp_state);\n')

        output_file.write('\t\t\t}\n')

        # (2) Create the transformation layer
        output_file.write('\t\t\ttemp_state = apply_gru(temp_state, transformed, state, &rnn_cell, FIXED_POINT_PRECISION);\n')
        output_file.write('\t\t\tstate = matrix_replace(state, temp_state);\n')
        output_file.write('\t\t\tmatrix_replace(prev_states[j], state);\n')
        output_file.write('\t\t}\n\n')

        # (3) Create the output layer
        output_layer = create_dense_layer('state', 'output', model_params['output'], prefix='\t\t')
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

        # Free all memory
        variables = ['transformed', 'state', 'fusion_gate', 'fusion_stack', 'output', 'temp_state']
        for variable in variables:
            output_file.write('\tmatrix_free({0});\n'.format(variable))

        output_file.write('\tfor (int16_t i = 0; i < SAMPLES_PER_SEQ; i++) {\n')
        output_file.write('\t\tmatrix_free(prev_states[i]);\n')
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
