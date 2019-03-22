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
