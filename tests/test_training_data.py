import numpy as np
import pandas as pd

from pyboretum import (
    TrainingData,
    MeanNode,
)


def test_training_data_turns_pandas_to_numpy(training_data_1d):
    X, Y = training_data_1d
    training_data = TrainingData(X, Y)

    assert isinstance(training_data.X, np.ndarray)
    assert isinstance(training_data.Y, np.ndarray)


def test_training_data_keeps_columns_and_index(training_data_1d):
    X, Y = training_data_1d
    training_data = TrainingData(X, Y)

    if isinstance(X, pd.DataFrame):
        assert list(training_data.X_names) == ['x']
        assert training_data.index.tolist() == X.index.tolist()
    else:
        assert list(training_data.X_names) == [0]
        assert training_data.index.tolist() == list(range(len(X)))


def test_training_data_from_numpy_defaults_indices(training_data_numpy):
    training_data = TrainingData(*training_data_numpy)

    assert training_data.index.tolist() == list(range(12))
    assert list(training_data.X_names) == [0]


def test_get_descendants(training_data_1d):
    X, Y = training_data_1d
    training_data = TrainingData(X, Y)

    node = MeanNode(X, Y, np.array([1.0]), 5.5)
    left, right = training_data.get_descendants(node)

    if isinstance(X, pd.DataFrame):
        assert left.index.tolist() == X.index[:6].tolist()
        assert right.index.tolist() == X.index[6:].tolist()
        assert list(left.X_names) == list(right.X_names) == list(training_data.X_names)
    else:
        assert left.index.tolist() == list(range(6))
        assert right.index.tolist() == list(range(6, 12))
        assert list(left.X_names) == list(right.X_names) == [0]


def test_1d_Y_is_transformed_to_2d():
    X = pd.DataFrame({'x': [1, 2, 3, 4]})
    y = pd.Series([1, 2, 3, 4])
    assert X.shape == (4, 1)
    assert y.shape == (4, )

    training_data = TrainingData(X, y)
    assert training_data.Y.shape == (4, 1)

    X = np.array([[1], [2], [3], [4]])
    y = np.array([1, 2, 3, 4])
    assert X.shape == (4, 1)
    assert y.shape == (4, )

    training_data = TrainingData(X, y)
    assert training_data.Y.shape == (4, 1)
