# standardlib
import csv
import re
from pathlib import Path, PosixPath
from typing import List, Tuple, Callable, Optional
from collections import namedtuple

# external package
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import numpy as np

# user
 
__all__ = ['stratify', 'random']

def stratify(x: np.ndarray, y: np.ndarray, train_size: int , random_state: int = 69):
    sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=1-train_size, random_state=random_state)
    Partition = namedtuple('Partition', ['x_train', 'y_train', 'x_test', 'y_test'])
    for train_index, test_index in sss.split(x, y):
        partition = Partition
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

    partition = Partition(x_train, y_train, x_test, y_test)
    return partition


def random(x: np.ndarray, y: np.ndarray, train_size: int, random_state: int = 69):
    Partition = namedtuple('Partition', ['x_train', 'y_train', 'x_test', 'y_test'])
    x_train, x_test, y_train, y_test = train_test_split(
                                                        x, 
                                                        y, 
                                                        train_size = train_size,
                                                        test_size = 1 - train_size, 
                                                        random_state = random_state)

    partition = Partition(x_train, y_train, x_test, y_test)
    return partition