import re
import tensorflow as tf
from google.protobuf import text_format
from typing import Dict, Any, Optional, List


GRADIENT_NAMES = ['gradient', 'adam', 'global_norm', 'exponentialdecay']
FILTER_NAMES = ['init', 'dropout'] + GRADIENT_NAMES
FILTER_OPS = ['NoOp', 'SparseSoftmaxCrossEntropyWithLogits', 'IsFinite', 'Squeeze', 'Shape', 'Const', 'Assign', 'Identity', 'GatherV2']
KEEP_OPS = ['MatMul', 'Add', 'Tanh', 'Sigmoid', 'BiasAdd', 'Softmax', 'Mul', 'Sub']
valid_input_types = ['Placeholder', 'VariableV2']


def parse_graph(graph_path: str) -> tf.GraphDef:
    """
    Parses the given protobuf text file into a Tensorflow GraphDef object.
    """
    with open(graph_path, 'r') as graph_file:
        graph_file_txt = graph_file.read()

    return text_format.Parse(graph_file_txt, tf.GraphDef())


def collect_nodes(graph_def: tf.GraphDef) -> Dict[str, tf.NodeDef]:
    """
    Creates a lookup table of node names to node definitions.
    """
    graph_nodes: Dict[str, tf.NodeDef] = dict()

    for node in graph_def.node:
        graph_nodes[node.name] = node

    return graph_nodes


def fetch_variables(nodes: Dict[str, tf.NodeDef]) -> Dict[str, tf.NodeDef]:
    result: Dict[str, tf.NodeDef] = dict()
    for name, node in nodes.items():
        if node.op in valid_input_types:
            result[name] = node

    return result


def create_variable_name(node_name: str) -> str:
    tokens = re.split(r'[\-/_]+', node_name)
    tokens = [t.capitalize() if i > 0 else t for i, t, in enumerate(tokens) if len(t) > 0]
    return ''.join(tokens)
