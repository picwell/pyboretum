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

    def get_descendants(self, node):
        mask = node.should_take_left(self.X)
        left_data = TrainingData(self.X[mask, :],
                                 self.Y[mask, :],
                                 self.index[mask],
                                 self.X_names)

        mask = ~mask
        right_data = TrainingData(self.X[mask, :],
                                  self.Y[mask, :],
                                  self.index[mask],
                                  self.X_names)

        return left_data, right_data


class ObliqueTrainingData(TrainingData):
    def __init__(self, X, Y, index=None, X_names=None, Y_names=None, oblique_features=None):
        """
        :param oblique_features: either the integer index of columns to include in oblique cuts, or, if X is a
        dataframe, can pass column names.
        """
        super(ObliqueTrainingData, self).__init__(X, Y, index, X_names, Y_names)

        if isinstance(X, pd.DataFrame):
            missing_columns = [feat for feat in oblique_features if feat not in X.columns]
            if missing_columns:
                raise ValueError('Following columns are missing from X: {}'.format(missing_columns))

            self.oblique_idxs = [X.columns.get_loc(col) for col in oblique_features]
        else:
            self.oblique_idxs = oblique_features