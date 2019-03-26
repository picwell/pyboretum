from graphviz import Digraph
import numpy as np
import pandas as pd

from pyboretum.splitters import MSESplitter
from pyboretum.tree import LinkedTree
from pyboretum.node import MeanNode
from pyboretum.training_data import TrainingData


class DecisionTree(object):
    def __init__(self, tree_class=LinkedTree, node_class=MeanNode,
                 max_depth=float('inf'), min_samples_leaf=1):
        self.tree_class = tree_class
        self.node_class = node_class

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        # These are initialized by .fit()
        self.pred_str = None
        self.tree = None
        self.X_names = None
        self.Y_names = None

    def _build_node(self, splitter, training_data):
        coeffs, threshold, cost = splitter.select_feature_to_cut(training_data.X,
                                                                 training_data.Y,
                                                                 self.min_samples_leaf)

        # TODO: this needs to be enabled later when statistical tests are used to select variables.
        # if threshold is None:
        #     threshold, cost = self.splitter.get_best_cutpoint()

        node = self.node_class(training_data.X, training_data.Y, coeffs, threshold, training_data.index)

        return node

    def _fit_core(self, splitter, node_id, node, training_data, depth):
        if depth == self.max_depth or node.coeffs is None:
            return
        else:
            left_data, right_data = training_data.get_descendants(node)

            left_node = self._build_node(splitter, left_data)
            right_node = self._build_node(splitter, right_data)

            left_id, right_id = self.tree.insert_children(node_id, left_node, right_node)

            self._fit_core(splitter, left_id, left_node, left_data, depth + 1)
            self._fit_core(splitter, right_id, right_node, right_data, depth + 1)

    def fit(self, X, Y, splitter=None):
        """
        How to do this without recursion??? where to store split feature/values? how to keep track of filtered X
        :param X:
        :param Y:
        :return:
        """
        training_data = TrainingData(X, Y)
        splitter = MSESplitter() if splitter is None else splitter

        self.X_names = training_data.X_names
        self.Y_names = training_data.Y_names

        # TODO: how can we ensure that the information used by pred_str is defined in the node?
        self.pred_str = splitter.pred_str

        root_node = self._build_node(splitter, training_data)
        self.tree = self.tree_class(root_node)

        self._fit_core(splitter, self.tree.get_root_id(), root_node, training_data, 0)

    def _get_leaf_node(self, x_row):
        """
        :param x_row: a single observation (row of X)
        :return: the ID and node object of the leaf
        """
        iterator = self.tree.get_iterator()
        while not iterator.is_leaf():
            node, _ = iterator.get_node()
            branch = node.which_branch(x_row)

            if branch == 'left':
                iterator.left_child()
            else:
                iterator.right_child()

        # current node is now a leaf node
        node, _ = iterator.get_node()
        return iterator.get_id(), node

    def apply(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        node_ids = []
        for row in range(X.shape[0]):
            node_id, _ = self._get_leaf_node(X[row, :])
            node_ids.append(node_id)

        return node_ids

    def predict(self, X, pred_str=None):
        if isinstance(X, pd.DataFrame):
            # Match column order with the training data:
            X_np = X[self.X_names].values
        else:
            X_np = X

        # Override the default pred_str:
        # TODO: how can we ensure that the information used by pred_str is defined in the node?
        pred_str = pred_str or self.pred_str

        predicted = []
        for row in range(X.shape[0]):
            _, node = self._get_leaf_node(X_np[row, :])
            predicted.append(getattr(node, pred_str))

        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(predicted, columns=self.Y_names, index=X.index)
        else:
            return np.array(predicted)

    def _get_nodes_and_edges(self, node_id, max_depth):
        node, depth = self.tree.get_node(node_id)

        if node.is_leaf() or depth >= max_depth:
            return [node_id], []
        else:
            left_id, right_id = self.tree.get_children_ids(node_id)
            left_nodes, left_edges = self._get_nodes_and_edges(left_id, max_depth)
            right_nodes, right_edges = self._get_nodes_and_edges(right_id, max_depth)

            nodes = [node_id] + left_nodes + right_nodes
            edges = ([(node_id, left_id, '<='), (node_id, right_id, '>')] +
                     left_edges + right_edges)
            return nodes, edges

    def get_nodes_and_edges(self, max_depth):
        nodes, edges = self._get_nodes_and_edges(self.tree.get_root_id(), max_depth)

        return nodes, edges

    def visualize_tree(self, max_depth=float('inf')):
        dot = Digraph(graph_attr=dict(size="12,12"))
        nodes, edges = self.get_nodes_and_edges(max_depth)
        for node_id in nodes:
            node, depth = self.tree.get_node(node_id)
            shape = 'cylinder'
            if node.is_leaf():
                color = 'palegreen3'
                shape = 'egg'
            elif depth == max_depth:
                color = 'burlywood3'
            else:
                color = 'burlywood'
            dot.node(str(node_id),
                     label=node.get_label(self.pred_str, self.X_names),
                     style='filled',
                     shape=shape,
                     color=color)

        for (e1, e2, label) in edges:
            dot.edge(str(e1),
                     str(e2),
                     label=label,
                     fontsize='25')

        return dot
