import pandas as pd

from pyboretum import (
    splitters,
    DecisionTree,
)


def test_get_node_label(training_data_1d):
    X, y = training_data_1d

    if isinstance(X, pd.DataFrame):
        feature_name = X.columns[0]
    else:
        feature_name = 0

    tree = DecisionTree(min_samples_leaf=2)
    tree.fit(X, y, splitter=splitters.MSESplitter())

    root_id = tree.tree.get_root_id()

    node, _ = tree.tree.get_node(root_id)
    assert (tree._get_node_label(node) ==
            'N Samples: 12\nAvg y: $[16.75]\n\nFeature: {}\nThreshold: 5.5'.format(feature_name))

    # Get to a leaf node:
    left_id, _ = tree.tree.get_children_ids(root_id)
    left_id, _ = tree.tree.get_children_ids(left_id)

    node, _ = tree.tree.get_node(left_id)
    assert (tree._get_node_label(node) == 'N Samples: 3\nAvg y: $[15.]')


def test_tree_returns_node_and_edges(training_data_1d):
    X, y = training_data_1d

    tree = DecisionTree(min_samples_leaf=2)
    tree.fit(X, y, splitter=splitters.MSESplitter())

    nodes, edges = tree.get_nodes_and_edges(max_depth=2)

    root_id = tree.tree.get_root_id()
    left_id, right_id = tree.tree.get_children_ids(root_id)
    left_left_id, left_right_id = tree.tree.get_children_ids(left_id)
    assert nodes[:4] == [root_id, left_id, left_left_id, left_right_id]
    assert edges[:4] == [(root_id, left_id, '<='), (root_id, right_id, '>'),
                         (left_id, left_left_id, '<='), (left_id, left_right_id, '>')]