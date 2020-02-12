import itertools
from collections.abc import MutableMapping

import pandas as pd

import cudf
from cudf.utils.utils import (
    NestedOrderedDict,
    OrderedColumnDict,
    cached_property,
)


class ColumnAccessor(MutableMapping):
    def __init__(self, data={}, multiindex=False, level_names=None):
        """
        Parameters
        ----------
        data : OrderedColumnDict (possibly nested)
        name : optional name for the ColumnAccessor
        multiindex : The keys convert to a Pandas MultiIndex
        """
        # TODO: we should validate the keys of `data`
        self._data = OrderedColumnDict(data)
        self.multiindex = multiindex
        if level_names is None:
            self.level_names = tuple((None,) * self.nlevels)
        else:
            self.level_names = tuple(level_names)

    def __iter__(self):
        return self._data.__iter__()

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self.set_by_label(key, value)
        self._clear_cache()

    def __delitem__(self, key):
        self._data.__delitem__(key)
        self._clear_cache()

    def __len__(self):
        return len(self._data)

    @property
    def nlevels(self):
        if not self.multiindex:
            return 1
        else:
            return len(next(iter(self.keys())))

    @property
    def name(self):
        return self.level_names[-1]

    @property
    def nrows(self):
        if len(self._data) == 0:
            return 0
        else:
            return len(next(iter(self.values())))

    @cached_property
    def names(self):
        return tuple(self.keys())

    @cached_property
    def columns(self):
        return tuple(self.values())

    @cached_property
    def _grouped_data(self):
        if self.multiindex:
            return NestedOrderedDict(zip(self.names, self.columns))
        else:
            return self._data

    def _clear_cache(self):
        cached_properties = "columns", "names", "_grouped_data"
        for attr in cached_properties:
            try:
                self.__delattr__(attr)
            except AttributeError:
                pass

    def to_pandas_index(self):
        if self.multiindex:
            result = pd.MultiIndex.from_tuples(
                self.names, names=self.level_names
            )
        else:
            result = pd.Index(
                self.names, name=self.level_names[0], tupleize_cols=False
            )
        return result

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
        self._data = self._data.__class__(zip(new_keys, new_values),)
        self._clear_cache()

    def copy(self):
        return self.__class__(
            self._data.copy(),
            multiindex=self.multiindex,
            level_names=self.level_names,
        )

    def get_by_label(self, key):
        if isinstance(key, slice):
            return self.get_by_label_slice(key)
        elif isinstance(key, list):
            return self.__class__(
                {k: self._data[k] for k in key},
                multiindex=self.multiindex,
                level_names=self.level_names,
            )
        else:
            result = self._grouped_data[key]
            if isinstance(result, cudf.core.column.ColumnBase):
                return self.__class__({key: result})
            else:
                result = _flatten(result)
                if not isinstance(key, tuple):
                    key = (key,)
                return self.__class__(
                    result,
                    multiindex=self.nlevels - len(key) > 1,
                    level_names=self.level_names[len(key) :],
                )

    def get_by_label_slice(self, key):
        start = key.start
        stop = key.stop
        if start is None:
            start = self.names[0]
        if stop is None:
            stop = self.names[-1]
        keys = itertools.chain(
            itertools.takewhile(
                lambda k: k != stop,
                itertools.dropwhile(lambda k: k != start, self.keys()),
            ),
            (stop,),
        )
        return self.__class__(
            {k: self._data[k] for k in keys},
            multiindex=self.multiindex,
            level_names=self.level_names,
        )

    def get_by_label_with_wildcard(self, key):
        return self.__class__(
            {k: self._data[k] for k in self._data if _compare_keys(k, key)},
            multiindex=self.multiindex,
            level_names=self.level_names,
        )

    def get_by_index(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self._data))
            keys = itertools.islice(self.keys(), start, stop)
        elif pd.api.types.is_integer(index):
            keys = itertools.islice(self.keys(), index, index + 1)
        else:
            keys = (self.names[i] for i in index)
        data = {k: self._data[k] for k in keys}
        return self.__class__(
            data, multiindex=self.multiindex, level_names=self.level_names,
        )

    def set_by_label(self, key, value):
        if self.multiindex:
            if not isinstance(key, tuple):
                key = (key,) + ("",) * (self.nlevels - 1)
        self._data[key] = value

    def _pad_key(self, key, pad_value=""):
        if not self.multiindex:
            return key
        if not isinstance(key, tuple):
            key = (key,)
        return key + (pad_value,) * (self.nlevels - len(key))


def _flatten(d):
    def _inner(d, parents=[]):
        for k, v in d.items():
            if not isinstance(v, d.__class__):
                if parents:
                    k = tuple(parents + [k])
                yield (k, v)
            else:
                yield from _inner(d=v, parents=parents + [k])

    return {k: v for k, v in _inner(d)}


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
