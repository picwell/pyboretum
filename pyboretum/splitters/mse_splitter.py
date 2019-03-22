from __future__ import absolute_import

import numpy as np

from .base import (
    Splitter,
    return_no_split,
)


class MSESplitter(Splitter):
    def __init__(self, covariance_matrix=None, *args, **kwargs):
        """
        :param covariance_matrix: an ndarray for covariance matrix
        """
        if covariance_matrix is None:
            # This is set when .get_best_cutpoint() is called:
            self.inverse_covariance_matrix = None

        else:
            self.inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

    @property
    def pred_str(self):
        return 'mean'

    def mahalanobis_distance(self, errors):
        """

        :param errors: a vector (1d or Nd) of errors (x_t - mu)
        :return: sum of mahalanobis distance of error(s)
        """
        return (errors * np.matmul(errors, self.inverse_covariance_matrix)).sum()

    def _get_binary_cutpoint(self, feature, y, min_samples_leaf):
        # This is more efficient than _get_ordered_cutpoint() since there is no sort involved.
        # The computational complexity is just O(n) where n is the number of samples.
        best_cutpoint = .5
        left_y = y[feature <= best_cutpoint]
        right_y = y[feature > best_cutpoint]
        if min(left_y.shape[0], right_y.shape[0]) > min_samples_leaf:
            left_errors = left_y - left_y.mean(axis=0)
            right_errors = right_y - right_y.mean(axis=0)
            mse = (self.mahalanobis_distance(left_errors) +
                   self.mahalanobis_distance(right_errors)) / y.shape[0]
        else:
            best_cutpoint, mse = return_no_split()

        return best_cutpoint, mse

    def _get_ordered_cutpoint(self, feature, y, min_samples_leaf):
        # The computational complexity is O(n*log(n)) determined by the sorting below.
        sorted_idx = np.argsort(feature)
        feature = feature[sorted_idx]
        y = y[sorted_idx]

        # See Torgo's thesis for more information about the implementation below.
        Sr, Nr = y.sum(axis=0), len(feature)

        Sl, Nl = np.zeros(y.shape[1]), 0
        bestTillNow = 0.0
        best_cutpoint, mse = return_no_split()
        for i in range(feature.shape[0] - 1):
            Sl += y[i]
            Sr -= y[i]
            Nl += 1
            Nr -= 1
            if (feature[i + 1] > feature[i]) and (min(Nl, Nr) >= min_samples_leaf):
                newSplitValue = (self.mahalanobis_distance(np.array([Sl])) / Nl) + \
                                (self.mahalanobis_distance(np.array([Sr])) / Nr)
                if newSplitValue > bestTillNow:
                    bestTillNow = newSplitValue
                    best_cutpoint = (feature[i] + feature[i + 1]) / 2.

        if best_cutpoint is not None:
            left_y = y[feature <= best_cutpoint]
            right_y = y[feature > best_cutpoint]
            mse = (np.sum(np.square(left_y - left_y.mean())) + np.sum(np.square(right_y - right_y.mean()))) / y.shape[0]

        return best_cutpoint, mse

    def get_best_cutpoint(self, feature, y, min_samples_leaf, **kwargs):
        if self.inverse_covariance_matrix is None:
            self.inverse_covariance_matrix = np.identity(y.shape[1])

        return super(self.__class__, self).get_best_cutpoint(feature, y, min_samples_leaf, **kwargs)