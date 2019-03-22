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
