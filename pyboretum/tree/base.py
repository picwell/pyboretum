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
import abc


class Tree(object):
    """
    A class to manage the storage for binary decision trees. Binary decision trees always
    insert both children nodes if a parent node is split. As a result, there is a relation
    between the number of leave nodes and the number of internal nodes:

        (number of leave nodes) = (number of internal nodes) + 1
    """

    @abc.abstractmethod
    def get_root_id(self):
        """
        Returns the root node ID. In many cases, this is used to start a recursion.
        :return: a node ID
        """
        pass

    @abc.abstractmethod
    def get_iterator(self):
        """
        Returns a TreeIterator object that starts at the root node and traverses the tree.
        :return: a TreeIterator object
        """
        pass

    @abc.abstractmethod
    def get_node(self, node_id):
        """
        Returns the corresponding Node object and its depth in a tree when a node ID is
        given. If the ID is not found, it returns None.
        :param node_id: a node ID
        :return: (a Node object or None, int or None)
        """
        pass

    @abc.abstractmethod
    def get_children_ids(self, node_id):
        """
        Returns the two child IDs of a node. Two Nones are returned if
            (a) the node itself is not found; or
            (b) the found node is a leaf node.
        :param node_id: a node ID
        :return: a pair of node IDs or Nones
        """
        pass

    @abc.abstractmethod
    def insert_children(self, node_id, left_node, right_node):
        """
        Insert two nodes as children nodes of the given parent node. Insertion fails if the
        parent node already has children.
        :param node_id: a node ID for the parent
        :param left_node, right_node: Node objects to insert
        :return: a pair of child node IDs if successful
        """
        pass


class TreeIterator(object):
    """
    An iterator used to traverse a tree. It maintains a state to make the traversal efficient.
    """
    @abc.abstractmethod
    def left_child(self):
        """
        Update the iterator to move to the left child. An exception is raised if the current
        node is a leaf node.
        """
        pass

    @abc.abstractmethod
    def right_child(self):
        """
        Update the iterator to move to the right child. An exception is raised if the current
        node is a leaf node.
        """
        pass

    @abc.abstractmethod
    def is_leaf(self):
        """
        Check whether the current node is a leaf node.
        :return: True or False.
        """
        pass

    @abc.abstractmethod
    def get_node(self):
        """
        Returns the corresponding Node object and its depth.
        :return: (a Node object, int)
        """
        pass

    @abc.abstractmethod
    def get_id(self):
        """
        Returns the ID of the corresponding Node.
        :return: a node ID
        """
        pass
