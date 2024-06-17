# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from __future__ import annotations

import itertools
import sys
from collections import abc
from functools import cached_property, reduce
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from pandas.api.types import is_bool

import cudf
from cudf.core import column

if TYPE_CHECKING:
    from cudf._typing import Dtype
    from cudf.core.column import ColumnBase


class _NestedGetItemDict(dict):
    """A dictionary whose __getitem__ method accesses nested dicts.

    This class directly subclasses dict for performance, so there are a number
    of gotchas: 1) the only safe accessor for nested elements is
    `__getitem__` (all other accessors will fail to perform nested lookups), 2)
    nested mappings will not exhibit the same behavior (they will be raw
    dictionaries unless explicitly created to be of this class), and 3) to
    construct this class you _must_ use `from_zip` to get appropriate treatment
    of tuple keys.
    """

    @classmethod
    def from_zip(cls, data):
        """Create from zip, specialized factory for nesting."""
        obj = cls()
        for key, value in data:
            d = obj
            for k in key[:-1]:
                d = d.setdefault(k, {})
            d[key[-1]] = value
        return obj

    def __getitem__(self, key):
        """Recursively apply dict.__getitem__ for nested elements."""
        # As described in the pandas docs
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html#advanced-indexing-with-hierarchical-index  # noqa: E501
        # accessing nested elements of a multiindex must be done using a tuple.
        # Lists and other sequences are treated as accessing multiple elements
        # at the top level of the index.
        if isinstance(key, tuple):
            return reduce(dict.__getitem__, key, self)
        return super().__getitem__(key)


def _to_flat_dict_inner(d: dict, parents: tuple = ()):
    for k, v in d.items():
        if not isinstance(v, d.__class__):
            if parents:
                k = parents + (k,)
            yield (k, v)
        else:
            yield from _to_flat_dict_inner(d=v, parents=parents + (k,))


class ColumnAccessor(abc.MutableMapping):
    """
    Parameters
    ----------
    data : mapping
        Mapping of keys to column values.
    multiindex : bool, optional
        Whether tuple keys represent a hierarchical
        index with multiple "levels" (default=False).
    level_names : tuple, optional
        Tuple containing names for each of the levels.
        For a non-hierarchical index, a tuple of size 1
        may be passe.
    rangeindex : bool, optional
        Whether the keys should be returned as a RangeIndex
        in `to_pandas_index` (default=False).
    label_dtype : Dtype, optional
        What dtype should be returned in `to_pandas_index`
        (default=None).
    verify : bool, optional
        For non ColumnAccessor inputs, whether to verify
        column length and type
    """

    _data: "Dict[Any, ColumnBase]"
    multiindex: bool
    _level_names: Tuple[Any, ...]

    def __init__(
        self,
        data: Union[abc.MutableMapping, ColumnAccessor, None] = None,
        multiindex: bool = False,
        level_names=None,
        rangeindex: bool = False,
        label_dtype: Dtype | None = None,
        verify: bool = True,
    ):
        self.rangeindex = rangeindex
        self.label_dtype = label_dtype
        if data is None:
            data = {}
        # TODO: we should validate the keys of `data`
        if isinstance(data, ColumnAccessor):
            multiindex = multiindex or data.multiindex
            level_names = level_names or data.level_names
            self._data = data._data
            self.multiindex = multiindex
            self._level_names = level_names
            self.rangeindex = data.rangeindex
            self.label_dtype = data.label_dtype
        else:
            # This code path is performance-critical for copies and should be
            # modified with care.
            data = dict(data)
            if data and verify:
                result = {}
                # Faster than next(iter(data.values()))
                column_length = len(data[next(iter(data))])
                for k, v in data.items():
                    # Much faster to avoid the function call if possible; the
                    # extra isinstance is negligible if we do have to make a
                    # column from something else.
                    if not isinstance(v, column.ColumnBase):
                        v = column.as_column(v)
                    if len(v) != column_length:
                        raise ValueError("All columns must be of equal length")
                    result[k] = v
                self._data = result
            else:
                self._data = data

            self.multiindex = multiindex
            self._level_names = level_names

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, key: Any) -> ColumnBase:
        return self._data[key]

    def __setitem__(self, key: Any, value: Any):
        self.set_by_label(key, value)

    def __delitem__(self, key: Any):
        old_ncols = len(self._data)
        del self._data[key]
        new_ncols = len(self._data)
        self._clear_cache(old_ncols, new_ncols)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        type_info = (
            f"{self.__class__.__name__}("
            f"multiindex={self.multiindex}, "
            f"level_names={self.level_names})"
        )
        column_info = "\n".join(
            [f"{name}: {col.dtype}" for name, col in self.items()]
        )
        return f"{type_info}\n{column_info}"

    def _from_columns_like_self(
        self, columns: abc.Iterable[ColumnBase], verify: bool = True
    ):
        """
        Return a new ColumnAccessor with columns and the properties of self.

        Parameters
        ----------
        columns : iterable of Columns
            New columns for the ColumnAccessor.
        verify : bool, optional
            Whether to verify column length and type.
        """
        if sys.version_info.major >= 3 and sys.version_info.minor >= 10:
            data = zip(self.names, columns, strict=True)
        else:
            columns = list(columns)
            if len(columns) != len(self.names):
                raise ValueError(
                    f"The number of columns ({len(columns)}) must match "
                    f"the number of existing column labels ({len(self.names)})."
                )
            data = zip(self.names, columns)
        return type(self)(
            data=dict(data),
            multiindex=self.multiindex,
            level_names=self.level_names,
            rangeindex=self.rangeindex,
            label_dtype=self.label_dtype,
            verify=verify,
        )

    @property
    def level_names(self) -> Tuple[Any, ...]:
        if self._level_names is None or len(self._level_names) == 0:
            return tuple((None,) * max(1, self.nlevels))
        else:
            return self._level_names

    @property
    def nlevels(self) -> int:
        if len(self._data) == 0:
            return 0
        if not self.multiindex:
            return 1
        else:
            return len(next(iter(self.keys())))

    @property
    def name(self) -> Any:
        return self.level_names[-1]

    @cached_property
    def nrows(self) -> int:
        if len(self._data) == 0:
            return 0
        else:
            return len(next(iter(self.values())))

    @cached_property
    def names(self) -> Tuple[Any, ...]:
        return tuple(self.keys())

    @cached_property
    def columns(self) -> Tuple[ColumnBase, ...]:
        return tuple(self.values())

    @cached_property
    def _grouped_data(self) -> abc.MutableMapping:
        """
        If self.multiindex is True,
        return the underlying mapping as a nested mapping.
        """
        if self.multiindex:
            return _NestedGetItemDict.from_zip(zip(self.names, self.columns))
        else:
            return self._data

    def _clear_cache(self, old_ncols: int, new_ncols: int):
        """
        Clear cached attributes.

        Parameters
        ----------
        old_ncols: int
            len(self._data) before self._data was modified
        new_ncols: int
            len(self._data) after self._data was modified
        """
        cached_properties = ("columns", "names", "_grouped_data")
        for attr in cached_properties:
            try:
                self.__delattr__(attr)
            except AttributeError:
                pass

        # nrows should only be cleared if empty before/after the op.
        if (old_ncols == 0) ^ (new_ncols == 0):
            try:
                del self.nrows
            except AttributeError:
                pass

    def to_pandas_index(self) -> pd.Index:
        """Convert the keys of the ColumnAccessor to a Pandas Index object."""
        if self.multiindex and len(self.level_names) > 0:
            result = pd.MultiIndex.from_tuples(
                self.names,
                names=self.level_names,
            )
        else:
            # Determine if we can return a RangeIndex
            if self.rangeindex:
                if not self.names:
                    return pd.RangeIndex(
                        start=0, stop=0, step=1, name=self.name
                    )
                elif cudf.api.types.infer_dtype(self.names) == "integer":
                    if len(self.names) == 1:
                        start = self.names[0]
                        return pd.RangeIndex(
                            start=start, stop=start + 1, step=1, name=self.name
                        )
                    uniques = np.unique(np.diff(np.array(self.names)))
                    if len(uniques) == 1 and uniques[0] != 0:
                        diff = uniques[0]
                        new_range = range(
                            self.names[0], self.names[-1] + diff, diff
                        )
                        return pd.RangeIndex(new_range, name=self.name)
            result = pd.Index(
                self.names,
                name=self.name,
                tupleize_cols=False,
                dtype=self.label_dtype,
            )
        return result

    def insert(
        self, name: Any, value: Any, loc: int = -1, validate: bool = True
    ):
        """
        Insert column into the ColumnAccessor at the specified location.

        Parameters
        ----------
        name : Name corresponding to the new column
        value : column-like
        loc : int, optional
            The location to insert the new value at.
            Must be (0 <= loc <= ncols). By default, the column is added
            to the end.

        Returns
        -------
        None, this function operates in-place.
        """
        name = self._pad_key(name)

        old_ncols = len(self._data)
        if loc == -1:
            loc = old_ncols
        if not (0 <= loc <= old_ncols):
            raise ValueError(
                f"insert: loc out of bounds: must be  0 <= loc <= {old_ncols}"
            )
        # TODO: we should move all insert logic here
        if name in self._data:
            raise ValueError(f"Cannot insert '{name}', already exists")
        if loc == old_ncols:
            if validate:
                value = column.as_column(value)
                if old_ncols > 0 and len(value) != self.nrows:
                    raise ValueError("All columns must be of equal length")
            self._data[name] = value
        else:
            new_keys = self.names[:loc] + (name,) + self.names[loc:]
            new_values = self.columns[:loc] + (value,) + self.columns[loc:]
            self._data = self._data.__class__(zip(new_keys, new_values))
        self._clear_cache(old_ncols, old_ncols + 1)

    def copy(self, deep=False) -> ColumnAccessor:
        """
        Make a copy of this ColumnAccessor.
        """
        if deep or cudf.get_option("copy_on_write"):
            data = {k: v.copy(deep=deep) for k, v in self._data.items()}
        else:
            data = self._data.copy()
        return self.__class__(
            data=data,
            multiindex=self.multiindex,
            level_names=self.level_names,
            rangeindex=self.rangeindex,
            label_dtype=self.label_dtype,
            verify=False,
        )

    def select_by_label(self, key: Any) -> ColumnAccessor:
        """
        Return a subset of this column accessor,
        composed of the keys specified by `key`.

        Parameters
        ----------
        key : slice, list-like, tuple or scalar

        Returns
        -------
        ColumnAccessor
        """
        if isinstance(key, slice):
            return self._select_by_label_slice(key)
        elif pd.api.types.is_list_like(key) and not isinstance(key, tuple):
            return self._select_by_label_list_like(key)
        else:
            if isinstance(key, tuple):
                if any(isinstance(k, slice) for k in key):
                    return self._select_by_label_with_wildcard(key)
            return self._select_by_label_grouped(key)

    def get_labels_by_index(self, index: Any) -> tuple:
        """Get the labels corresponding to the provided column indices.

        Parameters
        ----------
        index : integer, integer slice, boolean mask,
            or list-like of integers
            The column indexes.

        Returns
        -------
        tuple
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self._data))
            return self.names[start:stop:step]
        elif pd.api.types.is_integer(index):
            return (self.names[index],)
        elif (bn := len(index)) > 0 and all(map(is_bool, index)):
            if bn != (n := len(self.names)):
                raise IndexError(
                    f"Boolean mask has wrong length: {bn} not {n}"
                )
            if isinstance(index, (pd.Series, cudf.Series)):
                # Don't allow iloc indexing with series
                raise NotImplementedError(
                    "Cannot use Series object for mask iloc indexing"
                )
            # TODO: Doesn't handle on-device columns
            return tuple(n for n, keep in zip(self.names, index) if keep)
        else:
            return tuple(self.names[i] for i in index)

    def select_by_index(self, index: Any) -> ColumnAccessor:
        """
        Return a ColumnAccessor composed of the columns
        specified by index.

        Parameters
        ----------
        key : integer, integer slice, boolean mask,
            or list-like of integers

        Returns
        -------
        ColumnAccessor
        """
        keys = self.get_labels_by_index(index)
        data = {k: self._data[k] for k in keys}
        return self.__class__(
            data,
            multiindex=self.multiindex,
            level_names=self.level_names,
        )

    def swaplevel(self, i=-2, j=-1):
        """
        Swap level i with level j.
        Calling this method does not change the ordering of the values.

        Parameters
        ----------
        i : int or str, default -2
            First level of index to be swapped.
        j : int or str, default -1
            Second level of index to be swapped.

        Returns
        -------
        ColumnAccessor
        """

        i = _get_level(i, self.nlevels, self.level_names)
        j = _get_level(j, self.nlevels, self.level_names)

        new_keys = [list(row) for row in self]
        new_dict = {}

        # swap old keys for i and j
        for n, row in enumerate(self.names):
            new_keys[n][i], new_keys[n][j] = row[j], row[i]
            new_dict.update({row: tuple(new_keys[n])})

        new_data = {new_dict[k]: v.copy(deep=True) for k, v in self.items()}

        # swap level_names for i and j
        new_names = list(self.level_names)
        new_names[i], new_names[j] = new_names[j], new_names[i]

        return self.__class__(
            new_data,
            multiindex=True,
            level_names=new_names,
        )

    def set_by_label(self, key: Any, value: Any, validate: bool = True):
        """
        Add (or modify) column by name.

        Parameters
        ----------
        key
            name of the column
        value : column-like
            The value to insert into the column.
        validate : bool
            If True, the provided value will be coerced to a column and
            validated before setting (Default value = True).
        """
        key = self._pad_key(key)
        if validate:
            value = column.as_column(value)
            if len(self._data) > 0 and len(value) != self.nrows:
                raise ValueError("All columns must be of equal length")

        old_ncols = len(self._data)
        self._data[key] = value
        new_ncols = len(self._data)
        self._clear_cache(old_ncols, new_ncols)

    def _select_by_label_list_like(self, key: Any) -> ColumnAccessor:
        # Might be a generator
        key = tuple(key)
        # Special-casing for boolean mask
        if (bn := len(key)) > 0 and all(map(is_bool, key)):
            if bn != (n := len(self.names)):
                raise IndexError(
                    f"Boolean mask has wrong length: {bn} not {n}"
                )
            data = dict(
                item
                for item, keep in zip(self._grouped_data.items(), key)
                if keep
            )
        else:
            data = {k: self._grouped_data[k] for k in key}
        if self.multiindex:
            data = dict(_to_flat_dict_inner(data))
        return self.__class__(
            data,
            multiindex=self.multiindex,
            level_names=self.level_names,
        )

    def _select_by_label_grouped(self, key: Any) -> ColumnAccessor:
        result = self._grouped_data[key]
        if isinstance(result, column.ColumnBase):
            # self._grouped_data[key] = self._data[key] so skip validation
            return self.__class__(
                data={key: result},
                multiindex=self.multiindex,
                verify=False,
            )
        else:
            if self.multiindex:
                result = dict(_to_flat_dict_inner(result))
            if not isinstance(key, tuple):
                key = (key,)
            return self.__class__(
                result,
                multiindex=self.nlevels - len(key) > 1,
                level_names=self.level_names[len(key) :],
            )

    def _select_by_label_slice(self, key: slice) -> ColumnAccessor:
        start, stop = key.start, key.stop
        if key.step is not None:
            raise TypeError("Label slicing with step is not supported")

        if start is None:
            start = self.names[0]
        if stop is None:
            stop = self.names[-1]
        start = self._pad_key(start, slice(None))
        stop = self._pad_key(stop, slice(None))
        for idx, name in enumerate(self.names):
            if _keys_equal(name, start):
                start_idx = idx
                break
        for idx, name in enumerate(reversed(self.names)):
            if _keys_equal(name, stop):
                stop_idx = len(self.names) - idx
                break
        keys = self.names[start_idx:stop_idx]
        return self.__class__(
            {k: self._data[k] for k in keys},
            multiindex=self.multiindex,
            level_names=self.level_names,
            verify=False,
        )

    def _select_by_label_with_wildcard(self, key: Any) -> ColumnAccessor:
        key = self._pad_key(key, slice(None))
        return self.__class__(
            {k: self._data[k] for k in self._data if _keys_equal(k, key)},
            multiindex=self.multiindex,
            level_names=self.level_names,
            verify=False,
        )

    def _pad_key(self, key: Any, pad_value="") -> Any:
        """
        Pad the provided key to a length equal to the number
        of levels.
        """
        if not self.multiindex:
            return key
        if not isinstance(key, tuple):
            key = (key,)
        return key + (pad_value,) * (self.nlevels - len(key))

    def rename_levels(
        self, mapper: Union[Mapping[Any, Any], Callable], level: Optional[int]
    ) -> ColumnAccessor:
        """
        Rename the specified levels of the given ColumnAccessor

        Parameters
        ----------
        self : ColumnAccessor of a given dataframe

        mapper : dict-like or function transformations to apply to
            the column label values depending on selected ``level``.

            If dict-like, only replace the specified level of the
            ColumnAccessor's keys (that match the mapper's keys) with
            mapper's values

            If callable, the function is applied only to the specified level
            of the ColumnAccessor's keys.

        level : int
            In case of RangeIndex, only supported level is [0, None].
            In case of a MultiColumn, only the column labels in the specified
            level of the ColumnAccessor's keys will be transformed.

        Returns
        -------
        A new ColumnAccessor with values in the keys replaced according
        to the given mapper and level.

        """
        new_col_names: abc.Iterable
        if self.multiindex:

            def rename_column(x):
                x = list(x)
                if isinstance(mapper, Mapping):
                    x[level] = mapper.get(x[level], x[level])
                else:
                    x[level] = mapper(x[level])
                x = tuple(x)
                return x

            if level is None:
                raise NotImplementedError(
                    "Renaming columns with a MultiIndex and level=None is"
                    "not supported"
                )
            new_col_names = (rename_column(k) for k in self.keys())

        else:
            if level is None:
                level = 0
            if level != 0:
                raise IndexError(
                    f"Too many levels: Index has only 1 level, not {level+1}"
                )

            if isinstance(mapper, Mapping):
                new_col_names = [
                    mapper.get(col_name, col_name) for col_name in self.keys()
                ]
            else:
                new_col_names = [mapper(col_name) for col_name in self.keys()]

            if len(new_col_names) != len(set(new_col_names)):
                raise ValueError("Duplicate column names are not allowed")

        data = dict(zip(new_col_names, self.values()))
        return self.__class__(
            data=data,
            level_names=self.level_names,
            multiindex=self.multiindex,
            verify=False,
        )

    def droplevel(self, level):
        # drop the nth level
        if level < 0:
            level += self.nlevels

        old_ncols = len(self._data)
        self._data = {
            _remove_key_level(key, level): value
            for key, value in self._data.items()
        }
        new_ncols = len(self._data)
        self._level_names = (
            self._level_names[:level] + self._level_names[level + 1 :]
        )

        if (
            len(self._level_names) == 1
        ):  # can't use nlevels, as it depends on multiindex
            self.multiindex = False
        self._clear_cache(old_ncols, new_ncols)


def _keys_equal(target: Any, key: Any) -> bool:
    """
    Compare `key` to `target`.

    Return True if each value in `key` == corresponding value in `target`.
    If any value in `key` is slice(None), it is considered equal
    to the corresponding value in `target`.
    """
    if not isinstance(target, tuple):
        return target == key
    for k1, k2 in itertools.zip_longest(target, key, fillvalue=None):
        if k2 == slice(None):
            continue
        if k1 != k2:
            return False
    return True


def _remove_key_level(key: Any, level: int) -> Any:
    """
    Remove a level from key. If detupleize is True, and if only a
    single level remains, convert the tuple to a scalar.
    """
    result = key[:level] + key[level + 1 :]
    if len(result) == 1:
        return result[0]
    return result


def _get_level(x, nlevels, level_names):
    """Get the level index from a level number or name.

    If given an integer, this function will handle wraparound for
    negative values. If given a string (the level name), this function
    will extract the index of that level from `level_names`.

    Parameters
    ----------
    x
        The level number to validate
    nlevels
        The total available levels in the MultiIndex
    level_names
        The names of the levels.
    """
    if isinstance(x, int):
        if x < 0:
            x += nlevels
        if x >= nlevels:
            raise IndexError(
                f"Level {x} out of bounds. Index has {nlevels} levels."
            )
        return x
    else:
        x = level_names.index(x)
        return x
