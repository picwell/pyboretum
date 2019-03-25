import numpy as np

from pyboretum import (
    splitters,
    TrainingData,
)


def test_mae_splitter(training_data_1d):
    X, Y = training_data_1d
    training_data = TrainingData(X, Y)
    mae_splitter = splitters.MAESplitter()

    coeffs, cutpoint, cost = mae_splitter.select_feature_to_cut(training_data.X, training_data.Y, 2)

    index = np.argwhere(coeffs != 0.0)[0][0]
    assert training_data.X_names[index] in {'x', 0}
    # Brute force calculation reveals 8.5 results in best MAE reduction:
    assert cutpoint == 8.5


def test_min_samples_leaf_affects_mae_split(training_data_1d):
    X, Y = training_data_1d
    training_data = TrainingData(X, Y)
    mae_splitter = splitters.MAESplitter()

    coeffs, cutpoint, cost = mae_splitter.select_feature_to_cut(training_data.X, training_data.Y,
                                                                len(training_data.X) / 2)

    index = np.argwhere(coeffs != 0.0)[0][0]
    assert training_data.X_names[index] in {'x', 0}
    assert cutpoint == 5.5
