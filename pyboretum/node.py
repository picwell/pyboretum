import numpy as np


class Node(object):
    # A dictionary of attribute nomes to functions that calculate values for them. Override these
    # in the derived class.
    Y_FUNS = {}

    # Save indices when creating a node object:
    SAVE_IDS = False

    def __init__(self,  X, Y, feature, threshold, saved_ids=None, feature_name=None):
        self.n_samples = len(X)
        self.feature = feature
        self.threshold = threshold
        self.feature_name = feature_name

        if self.SAVE_IDS:
            self.saved_ids = saved_ids

        for key, fun in self.Y_FUNS.items():
            self.__dict__[key] = fun(Y, axis=0)

    def which_branch(self, x_row):
        """
        :param x_row: new sample (a row of X) to pass through the split point
        :return: 'left' or 'right' indicating decision at split point
            if feature is numeric: left means X feature value is < node split value
                                   right means X feature >= node split
            if feature is boolean: left means X[node.feature] is False, right -> True
            if feature is category: left means X[node.feature] in node.threshold
        """
        if x_row[self.feature] <= self.threshold:
            return 'left'
        else:
            return 'right'

    def is_leaf(self):
        return self.feature is None


# Sample Node implementations:
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
