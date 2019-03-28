# pyboretum
[![CircleCI](https://circleci.com/gh/picwell/pyboretum/tree/master.svg?style=svg)](https://circleci.com/gh/picwell/pyboretum/tree/master)

Fertile grounds to explore and analyze custom decision trees in Python

## Overview

All code in this package is licensed under the MIT License

## Getting Started

In this example, we'll use a small public dataset of red wine quality to demonstrate the basic pattern of training and inspecting a pyboretum decision tree.

```python
from pyboretum import DecisionTree, MeanNode

dt = DecisionTree(min_samples_leaf=5, max_depth=5,
				  node_class=MeanNode)
```

### Training a Decision Tree
Currently, pyboretum trees expect the data to be numeric (with plans to support string-encoded categorical features in the future).

#### Specifying a Splitter
When we fit a tree, in addition to passing `X` and `y` (our "features" and "target" data, respectively), we also specify a splitter (defaults to `MSESplitter`).  
Each splitter will partition the data to optimize a different objective, and this is where users can create their own custom splitters tailored to a particular problem at hand.

The currently offered splitters are all for regression tasks, although the interface can easily be applied to classification tasks (and we welcome splitter contributions if you decide to create your own!)

In the cells below, we will generate two different trees to minimize *mean squared error* and *mean absolute error*, two different splitters that are included out-of-the-box in pyboretum.

```python
import pandas as pd
data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
y = data['quality']
X[c for c in X.columns if c!='quality']

dt.fit(X, y, splitters.MSESplitter())
dt.visualize_tree(max_depth=2)
```
![MSE Tree](figures/wine_mse_tree.png)

We can pass a different splitter to `fit` to generate an alternative tree.

```python
from pyboretum import MedianNode

dt = DecisionTree(min_samples_leaf=5, max_depth=5,
				  node_class=MedianNode)
                  
dt.fit(X, y, splitters.MAESplitter())
dt.visualize_tree(max_depth=2)
```

![MSE Tree](figures/wine_mae_tree_v2.png)


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
