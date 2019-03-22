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


class LinkedTree(Tree):
    def __init__(self, node):
        self._root = _SubTree(node)

    def get_root_id(self):
        return self._root.get_id()

    def get_iterator(self):
        return _LinkedTreeIterator(self._root)

    def get_node(self, node_id):
        subtree, depth = self._root.find_subtree(node_id, 0)
        if subtree is None:
            return None, None
        else:
            return subtree.node, depth

    def get_children_ids(self, node_id):
        subtree, _ = self._root.find_subtree(node_id, 0)
        if subtree.left_subtree is None:
            return None, None

        else:
            return subtree.left_subtree.get_id(), subtree.right_subtree.get_id()

    def insert_children(self, node_id, left_node, right_node):
        subtree, _ = self._root.find_subtree(node_id, 0)
        assert subtree is not None, 'Node {} does not exist.'.format(node_id)
        assert subtree.left_subtree is None and subtree.right_subtree is None, \
            'Children nodes of {} already exist.'.format(node_id)

        subtree.left_subtree = _SubTree(left_node)
        subtree.right_subtree = _SubTree(right_node)

        return subtree.left_subtree.get_id(), subtree.right_subtree.get_id()


class _LinkedTreeIterator(TreeIterator):
    def __init__(self, root_subtree, depth=0):
        self._subtree = root_subtree
        self._depth = depth

    def left_child(self):
        assert self._subtree.left_subtree is not None, 'This is a leaf node.'
        self._subtree = self._subtree.left_subtree
        self._depth += 1

    def right_child(self):
        assert self._subtree.left_subtree is not None, 'This is a leaf node.'
        self._subtree = self._subtree.right_subtree
        self._depth += 1

    def is_leaf(self):
        return self._subtree.left_subtree is None

    def get_node(self):
        return self._subtree.node, self._depth

    def get_id(self):
        return self._subtree.get_id()


class _SubTree(object):
    def __init__(self, node):
        self.node = node
        self.left_subtree = None
        self.right_subtree = None

    def get_id(self):
        return id(self.node)

    def find_subtree(self, node_id, depth):
        # This assumes that an ID is unique, which is guaranteed by id().
        if self.get_id() == node_id:
            return self, depth

        elif self.left_subtree is None:
            # This is a leaf node.
            return None, None

        else:
            subtree, new_depth = self.left_subtree.find_subtree(node_id, depth + 1)
            # If it is not found on the left path, try right:
            if subtree is None:
                subtree, new_depth = self.right_subtree.find_subtree(node_id, depth + 1)

            return subtree, new_depth
