# pyboretum
[![CircleCI](https://circleci.com/gh/picwell/pyboretum/tree/master.svg?style=svg)](https://circleci.com/gh/picwell/pyboretum/tree/master)

Fertile grounds to explore and analyze custom decision trees in Python

## Overview

Mention the license

## Getting Started

Provide a short example to build an MAE tree and visualize

## Code Organization

```
<root_dir>/
  pyboretum/
   |-- splitters/
   |    |-- base.py (interface definition for Splitter)
   |    |-- mae_splitter.py (splitter for MAE criteria)
   |    |-- mse_splitter.py (splitter for MSE criteria, including using mahalanobis distance)
   |-- tree/
   |    |-- base.py (interface for Tree)
   |    |-- linked_tree.py (Tree implementation using linked lists)
   |    |-- list_tree.py (Tree implemenataion using lists)
   |-- decision_tree.py (main decidion tree implementation)
   |-- node.py (Node classes used with Tree)
   |-- training_data.py 
   |-- utils.py
  test/  (various unit tests)
```

## Key Classes/Interfaces for Customization

## What to Come Next

## License

This work is released under the MIT license agreement:
```
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
```

## Release Notes
