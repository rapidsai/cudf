import itertools
from abc import ABCMeta, abstractmethod, abstractproperty

from cudf.utils.utils import *


class ColumnAccessor(metaclass=ABCMeta):
    def __getitem__(self, key):
        return self.get_by_label(key)

    def __setitem__(self, key, value):
        self.set_by_label(key, value)

    @abstractmethod
    def get_by_label(self, key):
        """
        Get column or columns by key.
        """
        pass

    @abstractmethod
    def get_by_index(self, key):
        """
        Get columns specified by index.
        """
        pass

    @abstractmethod
    def get_by_label_range(self, key):
        """
        Get column or columns by a slice.
        """
        pass

    @abstractmethod
    def get_by_index_range(self, key):
        """
        Get column or columns by a slice of indexes.
        """
        pass

    @abstractmethod
    def set_by_label(self, key, value):
        pass

    @abstractproperty
    def names(self):
        pass

    @abstractproperty
    def values(self):
        pass


class OrderedDictColumnAccessor(ColumnAccessor):
    def __init__(self, data):
        """
        Parameters
        ----------
        data : OrderedColumnDict (possibly nested)
        """
        assert isinstance(data, OrderedColumnDict)
        self._data = data

    def get_by_label(self, key):
        return self._data[key]

    def get_by_label_range(self, key):
        return [
            self.get_by_label(k) for k in self._data if _compare_keys(k, key)
        ]

    def get_by_index(self, key):
        return next(itertools.islice(self._data.values(), key, key + 1))

    def get_by_index_range(self, start=0, end=None):
        if end is None:
            end = len(self._data)
        return list(itertools.islice(self._data.values(), start, end))

    def set_by_label(self, key, value):
        self._data[key] = value

    @property
    def names(self):
        return pd.Index(tuple(self._data.keys()))

    @property
    def values(self):
        return tuple(self._data.values())


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

    def get_by_label(self, key):
        return self._values[self._columns.get_loc(key)]

    def get_by_label_range(self, key):
        return self._values[self._columns.get_locs(key)]

    def get_by_index(self, index):
        return self._values[index]

    def get_by_index_range(self, start=0, end=None):
        if end is None:
            end = len(self._values)
        return self._values[start:end]

    def set_by_label(self, key, value):
        self._columns.append(key)
        self._values.append(value)

    @property
    def names(self):
        return self._columns

    @property
    def values(self):
        return tuple(self._values)


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
