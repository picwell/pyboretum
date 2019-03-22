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
from .base import (
    Tree,
    TreeIterator,
)
import math


def _get_left_index(node_index):
    return 2 * node_index + 1


def _get_right_index(node_index):
    return 2 * node_index + 2


def _get_depth(node_index):
    """
    The indices in depth d is
        (# of nodes up to depths d - 1) <= indices <= (# of nodes up to depth d) - 1

    where
        (# of nodes up to depth d) = 2^(d + 1) - 1

    :param node_index:
    :return: int for depth
    """
    return int(math.log(node_index + 1, 2))


class ListTree(Tree):
    def __init__(self, node):
        self._array = [node]

    def get_root_id(self):
        # Root is always the first entry in the list.
        return 0

    def get_iterator(self):
        return _ListTreeIterator(self)

    def _is_memory_allocated(self, node_id):
        return node_id < len(self._array)

    def _does_node_exist(self, node_id):
        return (self._is_memory_allocated(node_id) and
                self._array[node_id] is not None)

    def get_node(self, node_id):
        if self._does_node_exist(node_id):
            return self._array[node_id], _get_depth(node_id)
        else:
            return None, None

    def get_children_ids(self, node_id):
        right_id = _get_right_index(node_id)
        if self._does_node_exist(right_id):
            return _get_left_index(node_id), right_id
        else:
            return None, None

    def insert_children(self, node_id, left_node, right_node):
        assert self._does_node_exist(node_id), 'Node {} does not exist.'.format(node_id)

        left_id, right_id = _get_left_index(node_id), _get_right_index(node_id)
        assert not self._does_node_exist(left_id) and not self._does_node_exist(right_id), \
            'Children nodes of {} already exist.'.format(node_id)

        if not self._is_memory_allocated(right_id):
            self._array += [None] * (right_id - len(self._array) + 1)

        self._array[left_id] = left_node
        self._array[right_id] = right_node

        return left_id, right_id


class _ListTreeIterator(TreeIterator):
    def __init__(self, tree, index=0):
        # This assumes that the Tree is not updating during traversal.
        self._tree = tree
        self._index = index

    def left_child(self):
        left_index = _get_left_index(self._index)
        assert self._tree._does_node_exist(left_index), 'This is a leaf node.'
        self._index = left_index

    def right_child(self):
        right_index = _get_right_index(self._index)
        assert self._tree._does_node_exist(right_index), 'This is a leaf node.'
        self._index = right_index

    def is_leaf(self):
        left_id = _get_left_index(self._index)
        return not self._tree._does_node_exist(left_id)

    def get_node(self):
        return self._tree._array[self._index], _get_depth(self._index)

    def get_id(self):
        return self._index