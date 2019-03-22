from pyboretum import (
    LinkedTree,
    MeanNode,
    splitters,  # MAE and MSE splitters
    DecisionTree,
    TrainingData,
)

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