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
from pyboretum import TrainingData
import numpy as np
import pandas as pd


def test_training_data_turns_pandas_to_numpy(training_data_1d):
    X, Y = training_data_1d
    training_data = TrainingData(X, Y)

    assert isinstance(training_data.X, np.ndarray)
    assert isinstance(training_data.Y, np.ndarray)


def test_training_data_keeps_columns_and_index(training_data_1d):
    X, Y = training_data_1d
    training_data = TrainingData(X, Y)

    if isinstance(X, pd.DataFrame):
        assert training_data.X_names == ['x']
        assert training_data.index.tolist() == X.index.tolist()
    else:
        assert training_data.X_names == [0]
        assert training_data.index.tolist() == range(len(X))


def test_training_data_from_numpy_defaults_indices(training_data_numpy):
    training_data = TrainingData(*training_data_numpy)

    assert training_data.index.tolist() == range(12)
    assert training_data.X_names == [0]


def test_get_descendants(training_data_1d):
    X, Y = training_data_1d
    training_data = TrainingData(X, Y)
    left, right = training_data.get_descendants(0, 5.5)

    if isinstance(X, pd.DataFrame):
        assert left.index.tolist() == X.index[:6].tolist()
        assert right.index.tolist() == X.index[6:].tolist()
        assert left.X_names == right.X_names == training_data.X_names
    else:
        assert left.index.tolist() == range(6)
        assert right.index.tolist() == range(6, 12)
        assert left.X_names == right.X_names == [0]


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
