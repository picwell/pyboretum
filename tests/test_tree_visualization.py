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
from pyboretum import (
    splitters,  # MAE and MSE splitters
    DecisionTree,
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