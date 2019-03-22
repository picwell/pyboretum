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


def test_2d_predict_returns_correct_dimensions(training_data_mrt):
    X, Y = training_data_mrt

    tree = DecisionTree(max_depth=1)
    tree.fit(X, Y)

    # Checking the shape of the returned value:
    preds = tree.predict(X[:5])
    assert preds.shape == (5, Y.shape[1])
