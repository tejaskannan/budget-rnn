import tensorflow as tf
from collections import defaultdict
from typing import Dict, DefaultDict, Iterable, Optional, Set, List

from graph_utils import collect_nodes, fetch_variables, GRADIENT_NAMES


class ComputationGraph:

    def __init__(self, graph_def: tf.GraphDef):
        # Maps in which to store data
        self._node_id_map: Dict[str, int] = dict()  # Maps node name to node id
        self._id_node_map: Dict[int, str] = dict()  # Maps node id to node name
        self._node_data: Dict[int, tf.NodeDef] = dict()  # Maps node id to node data
        self._adj_list: DefaultDict[int, Set[int]] = defaultdict(set)  # Adjacency List
        self._rev_adj_list: DefaultDict[int, Set[int]] = defaultdict(set)  # Reverse Adjacency list

        # Create the graph object from this graph definition
        nodes = collect_nodes(graph_def)
        # graph_elements = fetch_variables(nodes)

        node_id = 0
        for node in graph_def.node:
            node_name = node.name

            if any((filter_word in node_name.lower() for filter_word in GRADIENT_NAMES)):
                continue

            # Add the node to the graph
            self._node_id_map[node_name] = node_id
            self._id_node_map[node_id] = node_name
            self._node_data[node_id] = node

            # Build up the adjacency list if possible
            for input_name in node.input:
                if input_name in self._node_id_map:
                    input_node_id = self._node_id_map[input_name]
                    self._adj_list[input_node_id].add(node_id)
                    self._rev_adj_list[node_id].add(input_node_id)

            node_id += 1

    def __len__(self) -> int:
        return len(self._node_data)

    def iterate_nodes(self) -> Iterable[tf.NodeDef]:
        node_ids = list(sorted(self._node_data.keys()))
        for node_id in node_ids:
            if node_id in self._node_data:
                yield self._node_data[node_id]

    def get_node_with_id(self, node_id: int) -> Optional[tf.NodeDef]:
        return self._node_data.get(node_id)

    def get_node_with_name(self, name: str) -> Optional[tf.NodeDef]:
        node_id = self._node_id_map.get(name)
        return self._node_data.get(node_id)

    def contains_node(self, name: str) -> bool:
        return name in self._node_id_map

    def get_next_nodes(self, node: tf.NodeDef) -> List[tf.NodeDef]:
        node_id = self._node_id_map[node.name]

        next_nodes: List[tf.NodeDef] = []
        for child_id in self._adj_list[node_id]:
            next_nodes.append(self._node_data[child_id])

        return next_nodes

    def get_prev_nodes(self, node: tf.NodeDef) -> List[tf.NodeDef]:
        node_id = self._node_id_map[node.name]

        next_nodes: List[tf.NodeDef] = []
        for parent_id in self._rev_adj_list[node_id]:
            next_nodes.append(self._node_data[parent_id])

        return next_nodes

    def prune_node(self, node: tf.NodeDef):
        node_id = self._node_id_map[node.name]
        previous_nodes = self.get_prev_nodes(node)

        # Can only prune nodes with at most one predecessor
        if len(previous_nodes) > 1:
            return

        # Remove the current node from the adjacency lists of the previous and next nodes
        # and re-arrange the edges to account for the pruning
        next_nodes = self.get_next_nodes(node)

        if len(previous_nodes) == 1:
            prev_node_id = self._node_id_map[previous_nodes[0].name]
            self._adj_list[prev_node_id].remove(node_id)

        for next_node in next_nodes:
            next_node_id = self._node_id_map[next_node.name]
            self._rev_adj_list[next_node_id].remove(node_id)

            if len(previous_nodes) == 1:
                prev_node_id = self._node_id_map[previous_nodes[0].name]
                self._rev_adj_list[next_node_id].add(prev_node_id)
                self._adj_list[prev_node_id].add(next_node_id)

        # Delete this node from all maps
        self._adj_list.pop(node_id)
        self._node_id_map.pop(node.name)
        self._id_node_map.pop(node_id)
        self._node_data.pop(node_id)
