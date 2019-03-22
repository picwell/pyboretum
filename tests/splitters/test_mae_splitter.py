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
from pyboretum import splitters, TrainingData


def test_mae_splitter(training_data_1d):
    X, Y = training_data_1d
    training_data = TrainingData(X, Y)
    mae_splitter = splitters.MAESplitter()

    feature, cutpoint, cost = mae_splitter.select_feature_to_cut(training_data.X, training_data.Y, 2)
    assert training_data.X_names[feature] in {'x', 0}
    # Brute force calculation reveals 8.5 results in best MAE reduction:
    assert cutpoint == 8.5


def test_min_samples_leaf_affects_mae_split(training_data_1d):
    X, Y = training_data_1d
    training_data = TrainingData(X, Y)
    mae_splitter = splitters.MAESplitter()

    feature, cutpoint, cost = mae_splitter.select_feature_to_cut(training_data.X, training_data.Y,
                                                                 len(training_data.X) / 2)

    assert training_data.X_names[feature] in {'x', 0}
    assert cutpoint == 5.5
