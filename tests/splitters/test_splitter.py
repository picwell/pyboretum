import pytest

from pyboretum import (
    splitters,
    TrainingData,
)


@pytest.mark.parametrize('splitter', [
    splitters.MSESplitter(),
    splitters.MAESplitter(),
])
def test_min_samples_leaf_can_stop_splitting(splitter, training_data_1d):
    X, Y = training_data_1d
    training_data = TrainingData(X, Y)

    feature, cutpoint, cost = splitter.select_feature_to_cut(training_data, len(training_data.X))

    assert feature == None
    assert cutpoint == None
    assert cost == float('inf')

