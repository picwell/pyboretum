import pytest
import pandas as pd
import numpy as np

from pyboretum import (
    Node,
    MeanNode,
    TrainingData,
)

@pytest.fixture()
def test_X():
    return pd.DataFrame([
        {'feature_0': 1, 'feature_1': 10.1},
        {'feature_0': 2, 'feature_1': 20.5},
        {'feature_0': 3, 'feature_1': 30.7},
    ], columns=['feature_0', 'feature_1'],
    index=['row_0', 'row_1', 'row_2'])


@pytest.fixture()
def test_Y():
    return pd.Series([1, 8, 3],
                     index=['row_0', 'row_1', 'row_2'])


def test_node_is_immutable(test_X, test_Y):
    node = MeanNode(test_X, test_Y, None, None)
    with pytest.raises(AttributeError) as e:
        node.n_samples = 100
    assert str(e.value) == "can't set attribute"


def test_node_can_evaluate_mean_and_median(test_X, test_Y):
    class BasicNode(Node):
        Y_FUNS={
            'mean': np.mean,
            'median': np.median,
        }

    training_data = TrainingData(test_X, test_Y)

    node = BasicNode(training_data.X, training_data.Y, np.array([1.0, 0.0]), 1.5)

    # Number of samples is always kept:
    assert node.n_samples == 3

    assert hasattr(node, 'mean')
    assert hasattr(node, 'median')

    assert node.mean == 4.0
    assert node.median == 3


def test_node_stores_row_indices_from_X(test_X, test_Y):
    node = MeanNode(test_X, test_Y, np.array([1.0, 0.0]), 1.5)
    assert node.n_samples == 3

    assert not hasattr(node, 'saved_ids')
    assert hasattr(node, 'mean')
    assert not hasattr(node, 'median')

    # TODO: can this definition be made simpler by building on top of MedianNode?
    class EnhancedNode(Node):
        Y_FUNS={
            'median': np.median,
        }
        SAVE_IDS=True

    training_data = TrainingData(test_X, test_Y)

    node = EnhancedNode(training_data.X, training_data.Y, np.array([1.0, 0.0]), 1.5, training_data.index)
    assert node.n_samples == 3

    assert hasattr(node, 'saved_ids')
    assert hasattr(node, 'median')

    assert node.saved_ids.tolist() == ['row_0', 'row_1', 'row_2']


def test_is_branch(test_X, test_Y):
    node = MeanNode(test_X, test_Y, np.array([1.0, 0.0]), 1.5)
    assert list(node.should_take_left(test_X.values)) == [True, False, False]

    node = MeanNode(test_X, test_Y, np.array([0.0, 1.0]), 25)
    assert list(node.should_take_left(test_X.values)) == [True, True, False]


def test_get_label(test_X, test_Y):
    # At leaf nodes:
    node = MeanNode(test_X, test_Y, None, None)
    assert node.get_label('mean', test_X.columns) == 'N Samples: 3\nAvg y: 4.0'

    # At internal nodes:
    node = MeanNode(test_X, test_Y, np.array([1.0, 0.0]), 2.5)
    assert (node.get_label('mean', test_X.columns) ==
            'N Samples: 3\nAvg y: 4.0\n\nFeature: {}\nThreshold: 2.5'.format([test_X.columns[0]]))

    node = MeanNode(test_X, test_Y, np.array([0.0, 1.0]), 3.5)
    assert (node.get_label('mean', test_X.columns) ==
            'N Samples: 3\nAvg y: 4.0\n\nFeature: {}\nThreshold: 3.5'.format([test_X.columns[1]]))
