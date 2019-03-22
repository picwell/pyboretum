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


def return_no_split():
    return None, float('inf')


class Splitter(object):
    @property
    def pred_str(self):
        """
        Value kept in nodes that should be used for prediction by default.
        """
        raise NotImplementedError()

    def _get_binary_cutpoint(self, feature, y, min_samples_leaf):
        """
        Default implementation. Override this implementation for better performance.
        """
        return self._get_ordered_cutpoint(feature, y, min_samples_leaf)

    def _get_ordered_cutpoint(self, feature, y, min_samples_leaf):
        raise NotImplementedError()

    def _get_unordered_cutpoint(self, feature, y, min_samples_leaf):
        raise NotImplementedError()

    def get_best_cutpoint(self, feature, y, min_samples_leaf, **kwargs):
        """
        :param X: dataframe of feature
        :param y: column name of feature we are trying to predict
        optional kwargs: min_samples_leaf, min_reduction
        :return:
        """

        # Is feature just binary indicator
        if (feature == 0).all():
            # Constant feature:
            best_cutpoint, cost = return_no_split()

        elif ((feature == 0) | (feature == 1)).all():
            best_cutpoint, cost = self._get_binary_cutpoint(feature, y, min_samples_leaf)

        else:
            # TODO: we have to think about how to work with nominal values.
            best_cutpoint, cost = self._get_ordered_cutpoint(feature, y, min_samples_leaf)

        return best_cutpoint, cost

    def select_feature_to_cut(self, X, y, min_samples_leaf):
        """
        This is a default implementation where the feature that gives the biggest gain when cut
        is chosen as the feature to cut.

        :param training_data: TrainingData object
        :param min_samples_leaf:
        :return: a tuple of (feature, cutpoint, cost). The cutpoint and cost can be None, and
                 float(inf), respectively, if the variable selection algorithm does not cut as well.
        """
        best_feature = None
        best_cutpoint, best_cost = return_no_split()

        for feature in range(X.shape[1]):
            cutpoint, cost = self.get_best_cutpoint(X[:, feature], y, min_samples_leaf)
            if cutpoint is not None and cost < best_cost:
                best_feature = feature
                best_cutpoint, best_cost = cutpoint, cost

        return best_feature, best_cutpoint, best_cost
