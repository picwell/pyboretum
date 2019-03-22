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
import numpy as np

from pyboretum import splitters, TrainingData


def test_mse_splitter_in_1d(training_data_1d):
    X, Y = training_data_1d
    training_data = TrainingData(X, Y)
    mse_splitter = splitters.MSESplitter()

    feature, cutpoint, cost = mse_splitter.select_feature_to_cut(training_data.X, training_data.Y, 1)

    assert training_data.X_names[feature] in {'x', 0}
    # Brute force calculation reveals 5.5 results in best MSE reduction:
    assert cutpoint == 5.5


def test_mse_splitter_in_2d(training_data_mrt):
    X, Y = training_data_mrt
    training_data = TrainingData(X, Y)
    mse_splitter = splitters.MSESplitter()

    feature, cutpoint, cost = mse_splitter.select_feature_to_cut(training_data.X, training_data.Y, 1)

    assert training_data.X_names[feature] in {'x', 0}
    # Brute force revealed X <= 13.5 gives best MSE reduction
    assert cutpoint == 13.5


def test_mse_splitter_in_2d_with_covariance(training_data_mrt):
    X, Y = training_data_mrt
    training_data = TrainingData(X, Y)
    cov_matrix = np.cov(training_data.Y.T)
    mse_splitter = splitters.MSESplitter(cov_matrix)

    feature, cutpoint, cost = mse_splitter.select_feature_to_cut(training_data.X, training_data.Y, 1)

    assert training_data.X_names[feature] in {'x', 0}
    # Brute force revealed X <= 6.5 gives best MSE reduction
    assert cutpoint == 6.5
