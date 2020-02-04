import itertools
import random
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd

from cudf.utils.utils import cached_property


class NestedOrderedDict(OrderedDict):
    def __missing__(self, key):
        self[key] = NestedOrderedDict()
        return self[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            d = self
            for k in key[:-1]:
                d = d[k]
            d.__setitem__(key[-1], value)
        else:
            super().__setitem__(key, value)


class ColumnAccessor(metaclass=ABCMeta):
    @abstractmethod
    def get_column(self, key):
        """
        Get a single column by key.
        """
        pass

    @abstractmethod
    def get_columns(self, key):
        """
        Get multiple columns specified by `key`
        """
        pass


class OrderedDictColumnAccessor(ColumnAccessor):
    def __init__(self, data):
        """
        Parameters
        ----------
        data : NestedOrderedDict (possibly nested)
        """
        assert isinstance(data, NestedOrderedDict)
        self._data = data

    def get_column(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        d = self._data
        for k in key:
            d = d[k]
        return d

    def get_columns(self, key):
        return [
            self.get_column(k)
            for k in self._flat_keys
            if _compare_keys(k, key)
        ]

    @cached_property
    def _flat_keys(self):
        return tuple(_flatten_keys(self._data))


class PandasColumnAccessor(ColumnAccessor):
    def __init__(self, columns, values):
        """
        Parameters
        ----------
        columns : A pd.Index object
        values : list of Columns
        """
        self._values = values
        self._columns = columns

    def get_column(self, key):
        return self._values[self._columns.get_loc(key)]

    def get_columns(self, key):
        return self._values[self._columns.get_locs(key)]


def _compare_keys(key, target):
    """
    Compare `key` to `target`.

    Return True if each value in target == corresponding value in `key`.
    If any value in `target` is slice(None), it is considered equal
    to the corresponding value in `key`.
    """
    for k1, k2 in itertools.zip_longest(key, target, fillvalue=None):
        if k2 == slice(None):
            continue
        if k1 != k2:
            return False
    return True


def _flatten_keys(d, parents=[]):
    for k, v in d.items():
        if not isinstance(v, NestedOrderedDict):
            yield tuple(parents + [k])
        else:
            yield from _flatten_keys(v, parents + [k])


if __name__ == "__main__":
    LEVELS = 3
    COLUMNS = 1000
    ROWS = 10

    columns = pd.MultiIndex.from_tuples(
        tuple(
            set(
                tuple(
                    random.choice("abcdefghjklmnopqrstuv")
                    for i in range(LEVELS)
                )
                for j in range(COLUMNS)
            )
        )
    )

    COLUMNS = len(columns)

    df = pd.DataFrame(np.random.rand(ROWS, COLUMNS), columns=columns)

    odict = NestedOrderedDict()
    for idx, col in zip(columns.values, df.values.T):
        odict[idx] = col
    odict_accessor = OrderedDictColumnAccessor(odict)
    pd_accessor = PandasColumnAccessor(columns, df.values.T)
