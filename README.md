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

## Release Notes
