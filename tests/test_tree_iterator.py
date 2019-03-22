"""
MIT License

Copyright (c) 2019 Picwell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import pandas as pd
import pytest

from pyboretum import (
    LinkedTree,
    ListTree,
    MeanNode,
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

# TODO: need more tests here!!!


@pytest.mark.parametrize('tree_class', [
    LinkedTree,
    ListTree,
])
def test_complex_tree(tree_class, test_nodes):
    tree = tree_class(test_nodes[0])
    root_id = tree.get_root_id()

    left_node_id, right_node_id = tree.insert_children(root_id,
                                                       test_nodes[1], test_nodes[2])
    left_left_node_id, left_right_node_id = tree.insert_children(left_node_id,
                                                                 test_nodes[3], test_nodes[4])

    iterator = tree.get_iterator()
    node, depth = iterator.get_node()
    assert node is test_nodes[0]
    assert depth == 0
    assert not iterator.is_leaf()

    iterator.left_child()
    node, depth = iterator.get_node()
    assert node is test_nodes[1]
    assert depth == 1
    assert not iterator.is_leaf()

    iterator.right_child()
    node, depth = iterator.get_node()
    assert node is test_nodes[4]
    assert depth == 2
    assert iterator.is_leaf()

    with pytest.raises(AssertionError):
        iterator.left_child()

    with pytest.raises(AssertionError):
        iterator.right_child()
