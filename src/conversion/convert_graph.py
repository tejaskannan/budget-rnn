import tensorflow as tf
from argparse import ArgumentParser
from typing import Dict, Optional, List

from graph_node import ComputationGraph
from graph_utils import parse_graph, FILTER_NAMES, FILTER_OPS, KEEP_OPS, create_variable_name
from graph_writing import create_matmul, create_definition, create_add, create_activation, create_softmax_prediction



def convert_name(input_name: str) -> str:
    if input_name.endswith('Tensordot'):
        return input_name + '/MatMul'
    return input_name


def get_input(input_name: str, graph: ComputationGraph, graph_elements: Dict[str, tf.NodeDef]) -> Optional[tf.NodeDef]: 
    input_name = convert_name(input_name)
    if not graph.contains_node(input_name):
        return None

    node = graph.get_node_with_name(input_name)

    node_stack = []
    while node.name not in graph_elements:
        node_stack.extend(node.input)

        if len(node_stack) == 0:
            return None

        node_name = convert_name(node_stack.pop())

        while not graph.contains_node(node_name):
            if len(node_stack) == 0:
                return None
            node_name = convert_name(node_stack.pop())

        node = graph.get_node_with_name(node_name)

    if input_name == 'model/rnn/while/rnn/multi_rnn_cell/cell_0/transform-layer-cell-0/mul_1':
        print(node)

    return node


def get_input_nodes(node: tf.NodeDef, graph: ComputationGraph, graph_elements: Dict[str, tf.NodeDef]) -> List[tf.NodeDef]:
    result: List[tf.NodeDef] = []

    for input_name in node.input:
        input_node = get_input(input_name, graph, graph_elements)
        if input_node is not None:
            result.append(input_node)

    return result

def compress_graph(graph: ComputationGraph) -> ComputationGraph:
    for node in graph.iterate_nodes():
        if node.op in FILTER_OPS:
            graph.prune_node(node)
        elif any((filter_word in node.name.lower() for filter_word in FILTER_NAMES)):
            graph.prune_node(node)

    return graph


def write_graph(graph: ComputationGraph, output_name: str):
    graph_elements: Dict[str, tf.NodeDef] = dict()
    variable_names: Dict[str, str] = dict()

    placeholders: List[str] = []
    for node in graph.iterate_nodes():
        if node.op in ('Placeholder', 'VariableV2') and 'dropout' not in node.name.lower():
            variable_names[node.name] = create_variable_name(node.name)
            graph_elements[node.name] = node
            
            if node.op == 'Placeholder':
                placeholders.append(node.name)

    lines: List[str] = []

    # Create the method definition
    lines.append(create_definition('neural_network_model', argument_names=placeholders))

    for node in graph.iterate_nodes():
        if node.op in KEEP_OPS and not any((filter_word in node.name.lower() for filter_word in FILTER_NAMES)):
            input_nodes = get_input_nodes(node, graph, graph_elements)
            var_name = create_variable_name(node.name)

            # if node.name == 'model/rnn/while/rnn/multi_rnn_cell/cell_0/transform-layer-cell-0/mul_1':
                # print(node)
                # print(input_nodes)
                # print('==========')

            if len(input_nodes) > 0:
                graph_elements[node.name] = node
                variable_names[node.name] = var_name

                if node.op == 'MatMul':
                    transpose_a, transpose_b = node.attr['transpose_a'].b, node.attr['transpose_b'].b
               
                    if len(input_nodes) == 2:
                        # print(node)

                        first_mat = variable_names[input_nodes[0].name]
                        second_mat = variable_names[input_nodes[1].name]

                        lines.append(create_matmul(first_mat, second_mat, var_name, transpose_a, transpose_b))
                elif node.op == 'BiasAdd':
                    first_mat = variable_names[input_nodes[0].name]
                    second_mat = variable_names[input_nodes[1].name]
                    lines.append(create_add(first_mat, second_mat, var_name))
                elif node.op in ('Tanh', 'Sigmoid', 'Relu'):
                    mat = variable_names[input_nodes[0].name]
                    lines.append(create_activation(mat, var_name, node.op.lower()))
                elif node.op == 'Softmax':
                    mat = variable_names[input_nodes[0].name]
                    lines.append(create_softmax_prediction(mat, output_name))
                elif node.op == 'ConcatV2':
                    print(node)
               # elif node.op == 'Mul':
               #     print(input_nodes)
               #     print(node)
               #     print('==========')
               # elif node.op == 'Add':
               #     print(input_nodes)
               #     print(node)
               #     print('==========')
               # elif node.op == 'Sub':
               #     print(node)

    # Add the closing brace for the function
    lines.append('}')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--graph-file', type=str, required=True)
    args = parser.parse_args()

    graph_def = parse_graph(args.graph_file)
    graph = ComputationGraph(graph_def)
    write_graph(graph, 'output')
