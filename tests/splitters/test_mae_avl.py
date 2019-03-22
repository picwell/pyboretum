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
from pyboretum.splitters.mae_splitter import MAE_AVL


def test_mae_avl_keeps_balance_with_add_remove():
    y = range(500)
    left_tree = None
    right_tree = MAE_AVL(y)

    subtree_size = 250
    assert len(right_tree.less_than) == len(right_tree.greater_than) == subtree_size
    assert right_tree.median == np.median(y)

    for i in range(0, 490, 10):
        right_tree.remove(y[i:i+10])
        subtree_size -= 5
        assert len(right_tree.less_than) == len(right_tree.greater_than) == subtree_size

        if left_tree is None:
            left_tree = MAE_AVL(y[i:i+10])
        else:
            left_tree.add(y[i:i+10])

        assert len(left_tree.less_than) == len(left_tree.greater_than) == (250 - subtree_size)

    left_tree.add(y[-10:])
    assert left_tree.median == np.median(y)


def test_avl_left_tree_larger_with_odd_number_items():
    y = range(1, 100)
    mae_avl = MAE_AVL(y)

    # if odd number of items, less_than subtree should have 1 more item
    assert (len(mae_avl.less_than) == 50) and (len(mae_avl.greater_than) == 49)

    # confirm respected after removing items
    mae_avl.remove(range(90, 100))
    assert (len(mae_avl.less_than) == 45) and (len(mae_avl.greater_than) == 44)

    # confirm respected after adding items
    mae_avl.add(range(1, 11))
    assert (len(mae_avl.less_than) == 50) and (len(mae_avl.greater_than) == 49)