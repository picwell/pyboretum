from __future__ import absolute_import

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
