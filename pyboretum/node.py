import numpy as np


# The current implementation is similar to collections.namedtuple.
def get_node_class(class_name, X_funs=None, Y_funs=None, save_ids=False):
    """
    :param class_name: name of the class
    :param X_funs: a Dict of name to functions applied to X.
    :param y_funs: a Dict of name to functions applied to y.
    :param save_ids: save indices of X as IDs.
    :return: a class to summarize node properties.
    """
    X_funs = {} if X_funs is None else X_funs
    Y_funs = {} if Y_funs is None else Y_funs

    def _init(self, X, Y, feature, threshold, saved_ids=None, feature_name=None):
        self.n_samples = len(X)
        self.feature = feature
        self.threshold = threshold
        self.feature_name = feature_name

        if save_ids:
            self.saved_ids = saved_ids

        for key, fun in X_funs.items():
            self.__dict__[key] = fun(X)
            
        for key, fun in Y_funs.items():
            self.__dict__[key] = fun(Y, axis=0)

    def _which_branch(self, x_row):
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

    def _is_leaf(self):
        return self.feature is None

    return type(class_name,
                (object, ),
                {
                    '__init__': _init,
                    'which_branch': _which_branch,
                    'is_leaf': _is_leaf,
                })


# Sample Node implementations:
MeanNode = get_node_class('MeanNode', Y_funs={
    'mean': np.mean,
})


MedianNode = get_node_class('MedianNode', Y_funs={
    'median': np.median,
})

MeanMedianAnalysisNode = get_node_class('MeanMedianAnalysisNode', save_ids=True, Y_funs={
    'mean': np.mean,
    'median': np.median,
})
