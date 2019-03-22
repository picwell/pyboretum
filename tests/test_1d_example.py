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
import pytest
import pandas as pd

from pyboretum import (
    MeanNode,
    MedianNode,
    splitters,  # MAE and MSE splitters
    DecisionTree,
)


@pytest.mark.parametrize('splitter, node_class', [
    (splitters.MSESplitter(), MeanNode),
    (splitters.MAESplitter(), MedianNode),
])
def test_with_one_sample_nodes(splitter, node_class, training_data_1d):
    X, y = training_data_1d

    tree = DecisionTree(node_class=node_class,
                        min_samples_leaf=1)
    tree.fit(X, y, splitter=splitter)

    # All y should form their own leaves:
    preds = tree.predict(X)
    if isinstance(X, pd.DataFrame):
        assert list(preds.index) == list(X.index)
        if isinstance(y, pd.DataFrame):
            assert list(preds.columns) == list(y.columns)

        preds = preds[0].tolist()

    assert preds == pytest.approx(y.tolist(), abs=1e-1)
    assert len(set(tree.apply(X))) == len(X)


def test_mse_tree(training_data_1d):
    X, y = training_data_1d

    tree = DecisionTree(min_samples_leaf=2)
    tree.fit(X, y, splitter=splitters.MSESplitter())

    preds = tree.predict(X)
    if isinstance(X, pd.DataFrame):
        assert list(preds.index) == list(X.index)
        if isinstance(y, pd.DataFrame):
            assert list(preds.columns) == list(y.columns)

        preds = preds[0].tolist()

    assert preds == pytest.approx([15.0, 15.0, 15.0,
                                   25.0, 25.0, 25.0,
                                   10.0, 10.0, 10.0,
                                   17.0, 17.0, 17.0], abs=1e-1)

    # 4 leaves are formed:
    node_ids = tree.apply(X)
    assert len(set(node_ids)) == 4

    for group_id in range(4):
        begin_idx = 3*group_id
        end_idx = 3*group_id + 2
        assert len(set(node_ids[begin_idx:end_idx])) == 1


def test_mae_tree(training_data_1d):
    X, y = training_data_1d

    tree = DecisionTree(node_class=MedianNode,
                        min_samples_leaf=2)
    tree.fit(X, y, splitter=splitters.MAESplitter())

    preds = tree.predict(X)
    if isinstance(X, pd.DataFrame):
        assert list(preds.index) == list(X.index)
        if isinstance(y, pd.DataFrame):
            assert list(preds.columns) == list(y.columns)

        preds = preds[0].tolist()

    assert preds == pytest.approx([15.1, 15.1, 15.1,
                                   25.2, 25.2, 25.2,
                                   9.8, 9.8, 9.8,
                                   17.0, 17.0, 17.0], abs=1e-1)

    # 4 leaves are formed:
    node_ids = tree.apply(X)
    assert len(set(node_ids)) == 4

    for group_id in range(4):
        begin_idx = 3*group_id
        end_idx = 3*group_id + 2
        assert len(set(node_ids[begin_idx:end_idx])) == 1
