import numpy as np


class Node(object):
    # A dictionary of attribute names to functions that calculate values from Y. Override
    # this in the derived class.
    Y_FUNS = {}

    # Save indices when creating a node object:
    SAVE_IDS = False

    def __init__(self,  X, Y, coeffs, threshold, saved_ids=None):
        """
        Our decision rule should look like
           h = coeff[0]*feature[0] + coeff[1]*feature[1] + ... + coeff[p]*feature[p] <= threshold

        where h represents a hyperplane in p dimensions. For a general nonlinear cut, we should
        consider combinding this with kernel methods.

        :param X, Y: numpy matrices
        :param coeffs: numpy array or None
        :param threshold: float or None
        :param saved_ids: List of IDs
        """
        self.n_samples = len(X)
        self.coeffs = coeffs
        self.threshold = threshold

        if self.SAVE_IDS:
            self.saved_ids = saved_ids

        for key, fun in self.Y_FUNS.items():
            self.__dict__[key] = fun(Y, axis=0)

    def should_take_left(self, X):
        """
        if feature is numeric: left means X feature value is < node split value
                               right means X feature >= node split
        if feature is boolean: left means X[node.feature] is False, right -> True
        if feature is category: left means X[node.feature] in node.threshold

        :param X: numpy matrix
        :return: Boolean numpy array
        """
        return np.matmul(X, self.coeffs) <= self.threshold

    def is_leaf(self):
        return self.coeffs is None

    def get_label(self, pred_str, feature_names):
        label = 'N Samples: {}\nAvg y: {}'.format(self.n_samples,
                                                  getattr(self, pred_str))

        if not self.is_leaf():
            assert len(feature_names) == len(self.coeffs), \
                'The length of feature_names should match the dimension of features.'

            used_feature_names = [name for name, coeff in zip(feature_names, self.coeffs)
                                  if coeff != 0.0]
            label += '\n\nFeature: {}\nThreshold: {}'.format(used_feature_names,
                                                             self.threshold)

        return label

"""
Sample Node implementations:
"""


class MeanNode(Node):
    Y_FUNS = {
        'mean': np.mean,
    }


class MedianNode(Node):
    Y_FUNS={
        'median': np.median,
    }

class MeanMedianAnalysisNode(object):
    SAVE_IDS=True
    Y_FUNS={
        'mean': np.mean,
        'median': np.median,
    }
