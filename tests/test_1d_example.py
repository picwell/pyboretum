import pytest
import numpy as np
import pandas as pd

from pyboretum import (
    LinkedTree,
    MeanNode,
    MedianNode,
    splitters,  # MAE and MSE splitters
    DecisionTree,
    TrainingData,
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
