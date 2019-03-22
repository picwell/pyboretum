from __future__ import absolute_import

import math
import numpy as np
import operator
from sortedcontainers import SortedList

from .base import (
    Splitter,
    return_no_split,
)


class MAESplitter(Splitter):
    def __init__(self, *args, **kwargs):
        pass

    @property
    def pred_str(self):
        return 'median'

    def _get_binary_cutpoint(self, feature, y, min_samples_leaf):
        best_cutpoint = .5
        left_y = y[feature <= best_cutpoint]
        right_y = y[feature > best_cutpoint]
        if min(left_y.shape[0], right_y.shape[0]) >= min_samples_leaf:
            mae = (np.sum(np.abs(left_y - np.median(left_y))) + np.sum(np.abs(right_y - np.median(right_y)))) / y.shape[0]
        else:
            best_cutpoint, mae = return_no_split()

        return best_cutpoint, mae

    def _get_ordered_cutpoint(self, feature, y, min_samples_leaf):
        sorted_idx = np.argsort(feature)
        feature = feature[sorted_idx]
        y = y[sorted_idx]

        right = MAE_AVL(y)
        left = None

        # The overall complexity of the following loop is O(n*log(n)) where n is the number of
        # samples, as long as the cost of insertion and deletion into the MAE_AVL is O(log(n)).
        points_to_move = []
        best_split_value = float('inf')
        best_cutpoint, mae = return_no_split()
        for i in range(len(feature) - 1):
            points_to_move.append(y[i])
            if feature[i+1] > feature[i]:
                if left is None:
                    left = MAE_AVL(points_to_move)
                else:
                    left.add(points_to_move)
                right.remove(points_to_move)

                left_sad = (left.greater_than_sum - left.less_than_sum +
                            left.median * (left.less_than_items - left.greater_than_items))

                right_sad = (right.greater_than_sum - right.less_than_sum +
                             right.median * (right.less_than_items - right.greater_than_items))

                new_split_value = right_sad + left_sad

                if (new_split_value < best_split_value and
                            min(i+1, len(feature)-(i+1)) >= min_samples_leaf):
                    best_split_value = new_split_value
                    best_cutpoint = (feature[i+1] + feature[i]) / 2.

                points_to_move = []

        if best_cutpoint is not None:
            left_y = y[feature <= best_cutpoint]
            right_y = y[feature > best_cutpoint]
            mae = (np.sum(np.abs(left_y - np.median(left_y))) + np.sum(np.abs(right_y - np.median(right_y)))) / y.shape[0]

        return best_cutpoint, mae


class MAE_AVL(object):
    def __init__(self, y):
        y = sorted(y)
        self.median = np.median(y)

        idx = int(math.ceil(len(y) / 2.))

        self.less_than = SortedList()
        self.less_than.update(y[:idx])
        self.less_than_sum = sum(y[:idx])
        self.less_than_items = len(self.less_than)

        self.greater_than = SortedList()
        self.greater_than.update(y[idx:])
        self.greater_than_sum = sum(y[idx:])
        self.greater_than_items = len(self.greater_than)

    def _update(self, values, add_or_remove_fn, add_or_subtract_values):
        for value in values:
            if self.less_than and value <= self.less_than[-1]:
                add_or_remove_fn(self.less_than, value)
                self.less_than_sum = add_or_subtract_values(self.less_than_sum, value)
                self.less_than_items = add_or_subtract_values(self.less_than_items, 1)
            else:
                add_or_remove_fn(self.greater_than, value)
                self.greater_than_sum = add_or_subtract_values(self.greater_than_sum, value)
                self.greater_than_items = add_or_subtract_values(self.greater_than_items, 1)

        if len(self.less_than) > len(self.greater_than)+1:
            while len(self.less_than) > len(self.greater_than)+1:
                x = self.less_than.pop(index=-1)
                self.greater_than.add(x)
                self.less_than_sum -= x
                self.greater_than_sum += x
                self.less_than_items -= 1
                self.greater_than_items += 1

        elif len(self.greater_than) > len(self.less_than):
            while len(self.greater_than) > len(self.less_than):
                x = self.greater_than.pop(index=0)
                self.less_than.add(x)
                self.less_than_sum += x
                self.greater_than_sum -= x
                self.less_than_items += 1
                self.greater_than_items -= 1

        if len(self.less_than) > len(self.greater_than):
            self.median = self.less_than[-1]
        else:
            self.median = (self.less_than[-1] + self.greater_than[0]) / 2.

    def remove(self, values):
        self._update(values, SortedList.remove, operator.sub)

    def add(self, values):
        self._update(values, SortedList.add, operator.add)
