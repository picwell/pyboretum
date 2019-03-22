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
import pytest
import numpy as np
import pandas as pd


def _add_index(X, Y, param):
    if param == 'pandas_with_integer_index':
        X.index = range(len(X))
        Y.index = range(len(X))

    elif param == 'pandas_with_string_index':
        index = ['row_{}'.format(idx) for idx in range(len(X))]
        X.index = index
        Y.index = index

    elif param == 'numpy':
        pass

    else:
        raise ValueError('Unknown parameter: {}'.format(request.param))


# Simple function where f(x) is
#     (a) 15 for 0 <= x < 3
#     (b) 25 for 3 <= x < 6
#     (c) 10 for 6 <= x < 9
#     (d) 17 for 9 <= x < 12
#
# Note that the cut points are different depending on whether MSE or MAE is used
# as splitting criteria.
@pytest.fixture(params=[
    'pandas_with_integer_index',
    'pandas_with_string_index',
    'numpy',
])
def training_data_1d(request):
    X = pd.DataFrame({'x': [0, 1, 2,
                            3, 4, 5,
                            6, 7, 8,
                            9, 10, 11]})

    # Values are chosen such that the means exactly matches the truth:
    y = pd.Series([14.6, 15.1, 15.3,
                   24.3, 25.2, 25.5,
                   10.4, 9.8, 9.8,
                   17.0, 16.9, 17.1])

    _add_index(X, y, request.param)

    if request.param.startswith('pandas'):
        return X, y

    elif request.param.startswith('numpy'):
        return X.values, y.values


# For y1,
#     10 for 0 <= x <= 7
#     15 for 8 <= x <= 20
# For y2,
#     10 for 0 <= x <= 14
#     25 for 15 <= x <= 20
#
# The noise on y1 is an order of magnitude smaller than that on y2.
@pytest.fixture(params=[
    'pandas_with_integer_index',
    'pandas_with_string_index',
    'numpy',
])
def training_data_mrt(request):
    X = pd.DataFrame({'x': range(20)})

    Y = pd.DataFrame({
        'y1': [
            10.3, 9.8, 9.7, 10.5, 10.0, 9.9, 9.8,
            15.0, 15.1, 15.3, 14.6, 14.8, 14.8, 15.4,
            15.1, 15.1, 14.6, 15.3, 15.0, 14.9,
        ],
        'y2': [
            13.0, 8.0, 7.0, 15.0, 10.0, 9.0, 8.0,
            10.0, 11.0, 13.0, 6.0, 8.0, 8.0, 14.0,
            26.0, 26.0, 21.0, 28.0, 25.0, 24.0,
        ]})

    _add_index(X, Y, request.param)

    if request.param.startswith('pandas'):
        return X, Y

    elif request.param.startswith('numpy'):
        return X.values, Y.values


@pytest.fixture()
def training_data_numpy():
    X = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    Y = np.array([14.6, 15.1, 15.3,
                  24.3, 25.2, 25.5,
                  10.4, 9.8, 9.8,
                  17.0, 16.9, 17.1])

    return X, Y