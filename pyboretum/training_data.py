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
import pandas as pd


def enforce_matrix(array):
    if array.ndim == 1:
        return array.reshape(-1, 1)
    else:
        return array


class TrainingData(object):
    """
    A class for standardizing and storing Tree training data w/ following attributes:

        X: numpy array of underlying training features
        y: numpy array of dependent variable/output we are trying to predict
        index: the index/ID of the samples in X and y
        X_names: the names of features in X
        Y_names: the names of targets in y

    Can be initialized with either DataFrame or numpy array objects
    """
    def __init__(self, X, Y, index=None, X_names=None, Y_names=None):
        # TODO: should this work for any combinations of DataFrames and numpy matrices for X and Y?
        if isinstance(X, pd.DataFrame):
            # Sort X and y with pandas to match their indices:
            assert (X.index == Y.index).all(), 'X and y have different indices'

            if isinstance(Y, pd.Series):
                Y = Y.to_frame()

            self.index = np.array(X.index.tolist())
            self.X_names = X.columns
            self.Y_names = Y.columns

            # TODO: should we keep columns of X as np.array to make access faster?
            self.X = X.values
            self.Y = Y.values

        elif isinstance(X, np.ndarray):
            self.X = enforce_matrix(X)
            self.Y = enforce_matrix(Y)

            if index is None:
                self.index = np.array(range((X.shape[0])))
            else:
                self.index = index

            # If we are passed column names, use those
            self.X_names = range(self.X.shape[1]) if X_names is None else X_names
            assert len(self.X_names) == self.X.shape[1], 'X_names does not match in length with X.'

            self.Y_names = range(self.Y.shape[1]) if Y_names is None else Y_names
            assert len(self.Y_names) == self.Y.shape[1], 'Y_names does not match in length with Y.'

        else:
            raise TypeError("Input X must be pandas dataframe or numpy array.")

    def get_descendants(self, feature, threshold):
        left_mask = self.X[:, feature] <= threshold
        right_mask = ~left_mask

        left_data = TrainingData(self.X[left_mask], self.Y[left_mask], self.index[left_mask], self.X_names)
        right_data = TrainingData(self.X[right_mask], self.Y[right_mask], self.index[right_mask], self.X_names)

        return left_data, right_data