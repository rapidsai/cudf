import itertools
from collections.abc import MutableMapping

from cudf.utils.utils import OrderedColumnDict


class ColumnAccessor(MutableMapping):
    def __init__(self, data={}, name=None, multiindex=False):
        """
        Parameters
        ----------
        data : OrderedColumnDict (possibly nested)
        name : optional name for the ColumnAccessor
        multiindex : The keys convert to a Pandas MultiIndex
        """
        # TODO: we should validate the keys of `data`
        self._data = OrderedColumnDict(data)
        self.name = name
        self.multiindex = multiindex

    def __iter__(self):
        return self._data.__iter__()

    def __getitem__(self, key):
        return self.get_by_label(key)

    def __setitem__(self, key, value):
        self.set_by_label(key, value)

    def __delitem__(self, key):
        self._data.__delitem__(key)

    def __len__(self):
        return len(self._data)

    def insert(self, name, value, loc=-1):
        """
        Insert value at specified location.
        """
        # TODO: we should move all insert logic here
        name = self._pad_key(name)
        new_keys = list(self.keys())
        new_values = list(self.values())
        new_keys.insert(loc, name)
        new_values.insert(loc, value)
        self._data = self._data.__class__((zip(new_keys, new_values)))

    def copy(self):
        return self.__class__(self._data.copy(), multiindex=self.multiindex)

    def get_by_label(self, key):
        return self._data[key]

    def get_by_label_range(self, key):
        return [
            self.get_by_label(k) for k in self._data if _compare_keys(k, key)
        ]

    def get_by_index(self, key):
        return next(itertools.islice(self.values(), key, key + 1))

    def get_by_index_range(self, start=0, end=None):
        if end is None:
            end = len(self._data)
        return list(itertools.islice(self.values(), start, end))

    def set_by_label(self, key, value):
        if self.multiindex:
            if not isinstance(key, tuple):
                key = (key,) + ("",) * (self.nlevels - 1)
        self._data[key] = value

    @property
    def names(self):
        return tuple(self.keys())

    @property
    def columns(self):
        return tuple(self.values())

    @property
    def nlevels(self):
        if not self.multiindex:
            return 1
        else:
            return len(self.names[0])

    def _pad_key(self, key, pad_value=""):
        if not self.multiindex:
            return key
        if not isinstance(key, tuple):
            key = (key,)
        return key + (pad_value,) * (self.nlevels - 1)


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
