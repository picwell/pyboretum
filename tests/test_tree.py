import pandas as pd
import pytest

from pyboretum import (
    LinkedTree,
    ListTree,
    get_node_class,
    MeanNode,
    MedianNode,
)


@pytest.fixture()
def test_nodes():
    X = pd.DataFrame([
        {'feature_0': 1, 'feature_1': 10.1},
        {'feature_0': 2, 'feature_1': 20.5},
        {'feature_0': 3, 'feature_1': 30.7},
        {'feature_0': 4, 'feature_1': 40.2},
        {'feature_0': 5, 'feature_1': 50.8},
    ], columns=['feature_0', 'feature_1'],
    index=['row_0', 'row_1', 'row_2', 'row_3', 'row_4'])

    y = pd.Series([1, 8, 3, 10, 4],
                  index=['row_0', 'row_1', 'row_2', 'row_3', 'row_4'])

    return tuple(MeanNode(X.iloc[:(idx + 1)], y.iloc[:(idx + 1)], 'feature_0', 2.5)
                 for idx in range(5))


_NONEXISTING_ID = 1234567


# This function knows the implementation details for testing:
def _get_id(node):
    return id(node)


def test_get_root_id_returns_the_correct_node_for_linked_tree(test_nodes):
    node = test_nodes[0]
    tree = LinkedTree(node)

    # This test reveals implementation details:
    root_id = tree.get_root_id()
    assert root_id == _get_id(node)

    returned_node, depth = tree.get_node(root_id)
    assert returned_node is node
    assert depth == 0


def test_get_root_id_returns_the_correct_node_for_list_tree(test_nodes):
    node = test_nodes[0]
    tree = ListTree(node)

    # This test reveals implementation details:
    root_id = tree.get_root_id()
    assert root_id == 0

    returned_node, depth = tree.get_node(root_id)
    assert returned_node is node
    assert depth == 0


@pytest.mark.parametrize('tree_class', [
    LinkedTree,
    ListTree,
])
def test_get_node_returns_the_correct_node(tree_class, test_nodes):
    tree = tree_class(test_nodes[0])
    root_id = tree.get_root_id()

    returned_node, depth = tree.get_node(root_id)
    assert returned_node is test_nodes[0]
    assert depth == 0

    # A non-existing node:
    returned_node, depth = tree.get_node(_NONEXISTING_ID)
    assert returned_node is None
    assert depth is None


def test_get_node_returns_nones_when_the_node_is_not_initialized(test_nodes):
    """
    This test is only necessary for ListTree.
    """
    tree = ListTree(test_nodes[0])
    root_id = tree.get_root_id()

    left_node_id, right_node_id = tree.insert_children(root_id,
                                                       test_nodes[1], test_nodes[2])
    # This will allocate memory for the left subtree for ListTree:
    left_left_node_id, left_right_node_id = tree.insert_children(right_node_id,
                                                                 test_nodes[3], test_nodes[4])

    # These checks expose implementation details, but this ensures that the node ID used for
    # the actual test is valid.
    assert root_id == 0
    assert left_node_id == 1
    assert right_node_id == 2
    assert left_left_node_id == 5
    assert left_right_node_id == 6

    returned_node, depth = tree.get_node(3)
    assert returned_node is None
    assert depth is None


@pytest.mark.parametrize('tree_class', [
    LinkedTree,
    ListTree,
])
def test_insert_children_works(tree_class, test_nodes):
    tree = tree_class(test_nodes[0])
    root_id = tree.get_root_id()

    left_node_id, right_node_id = tree.insert_children(root_id, test_nodes[1], test_nodes[2])

    returned_node, depth = tree.get_node(left_node_id)
    assert returned_node is test_nodes[1]
    assert returned_node.n_samples == 2
    assert depth == 1

    returned_node, depth = tree.get_node(right_node_id)
    assert returned_node is test_nodes[2]
    assert returned_node.n_samples == 3
    assert depth == 1

    # Cannot insert again to the same node:
    with pytest.raises(AssertionError) as e:
        tree.insert_children(root_id, test_nodes[1], test_nodes[2])
    assert str(e.value) == 'Children nodes of {} already exist.'.format(root_id)

    # Cannot insert to a non-existing node:
    with pytest.raises(AssertionError) as e:
        tree.insert_children(_NONEXISTING_ID, test_nodes[3], test_nodes[4])
    assert str(e.value) == 'Node {} does not exist.'.format(_NONEXISTING_ID)


@pytest.mark.parametrize('tree_class', [
    LinkedTree,
    ListTree,
])
def test_get_children_ids(tree_class, test_nodes):
    tree = tree_class(test_nodes[0])
    root_id = tree.get_root_id()

    left_node_id, right_node_id = tree.get_children_ids(root_id)
    assert left_node_id is None
    assert right_node_id is None

    left_node_id, right_node_id = tree.insert_children(root_id, test_nodes[1], test_nodes[2])

    returned_left_node_id, returned_right_node_id = tree.get_children_ids(root_id)
    return returned_left_node_id == left_node_id
    return returned_right_node_id == right_node_id

    # Nones are returned if the node ID does not exist:
    left_node_id, right_node_id = tree.get_children_ids(_NONEXISTING_ID)
    assert left_node_id is None
    assert right_node_id is None


@pytest.mark.parametrize('tree_class', [
    LinkedTree,
    ListTree,
])
def test_complex_tree(tree_class, test_nodes):
    tree = tree_class(test_nodes[0])
    root_id = tree.get_root_id()

    left_node_id, right_node_id = tree.insert_children(root_id,
                                                       test_nodes[1], test_nodes[2])
    left_left_node_id, left_right_node_id = tree.insert_children(right_node_id,
                                                                 test_nodes[3], test_nodes[4])

    # Check depth 1:
    returned_left_node_id, returned_right_node_id = tree.get_children_ids(root_id)
    assert returned_left_node_id == left_node_id
    assert returned_right_node_id == right_node_id

    returned_node, depth = tree.get_node(returned_left_node_id)
    assert returned_node is test_nodes[1]
    assert depth == 1

    returned_node, depth = tree.get_node(returned_right_node_id)
    assert returned_node is test_nodes[2]
    assert depth == 1

    # Check depth 2 on the left side:
    returned_left_node_id, returned_right_node_id = tree.get_children_ids(left_node_id)
    assert returned_left_node_id is None
    assert returned_right_node_id is None

    # Check depth 2 on the right side:
    returned_left_node_id, returned_right_node_id = tree.get_children_ids(right_node_id)
    assert returned_left_node_id == left_left_node_id
    assert returned_right_node_id == left_right_node_id

    returned_node, depth = tree.get_node(returned_left_node_id)
    assert returned_node is test_nodes[3]
    assert depth == 2

    returned_node, depth = tree.get_node(returned_right_node_id)
    assert returned_node is test_nodes[4]
    assert depth == 2
