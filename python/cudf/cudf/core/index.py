# Copyright (c) 2018-2021, NVIDIA CORPORATION.

from __future__ import annotations, division, print_function

import pickle
from numbers import Number
from typing import Any, Dict, Optional, Tuple, Type, Union

import cupy
import numpy as np
import pandas as pd
from nvtx import annotate
from pandas._config import get_option

import cudf
from cudf._lib.filling import sequence
from cudf._lib.search import search_sorted
from cudf._lib.table import Table
from cudf._typing import DtypeObj
from cudf.api.types import (
    _is_scalar_or_zero_d_array,
    is_dtype_equal,
    is_integer,
    is_string_dtype,
)
from cudf.core.abc import Serializable
from cudf.core.column import (
    CategoricalColumn,
    ColumnBase,
    DatetimeColumn,
    IntervalColumn,
    NumericalColumn,
    StringColumn,
    TimeDeltaColumn,
    arange,
    column,
)
from cudf.core.column.column import _concat_columns, as_column
from cudf.core.column.string import StringMethods as StringMethods
from cudf.core.dtypes import IntervalDtype
from cudf.core.frame import SingleColumnFrame
from cudf.utils import ioutils
from cudf.utils.docutils import copy_docstring
from cudf.utils.dtypes import (
    _is_non_decimal_numeric_dtype,
    find_common_type,
    is_categorical_dtype,
    is_interval_dtype,
    is_list_like,
    is_mixed_with_object_dtype,
    is_scalar,
    numeric_normalize_types,
)
from cudf.utils.utils import cached_property, search_range


class BaseIndex(SingleColumnFrame, Serializable):
    """Base class for all cudf Index types."""

    dtype: DtypeObj

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        if method == "__call__" and hasattr(cudf, ufunc.__name__):
            func = getattr(cudf, ufunc.__name__)
            return func(*inputs)
        else:
            return NotImplemented

    @cached_property
    def _values(self) -> ColumnBase:
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError()

    def drop_duplicates(self, keep="first"):
        """
        Return Index with duplicate values removed

        Parameters
        ----------
        keep : {‘first’, ‘last’, False}, default ‘first’
            * ‘first’ : Drop duplicates except for the
                first occurrence.
            * ‘last’ : Drop duplicates except for the
                last occurrence.
            *  False : Drop all duplicates.

        Returns
        -------
        deduplicated : Index

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'])
        >>> idx
        StringIndex(['lama' 'cow' 'lama' 'beetle' 'lama' 'hippo'], dtype='object')
        >>> idx.drop_duplicates()
        StringIndex(['beetle' 'cow' 'hippo' 'lama'], dtype='object')
        """  # noqa: E501
        return super().drop_duplicates(keep=keep)

    def serialize(self):
        header = {}
        header["index_column"] = {}
        # store metadata values of index separately
        # Indexes: Numerical/DateTime/String are often GPU backed
        header["index_column"], frames = self._values.serialize()

        header["name"] = pickle.dumps(self.name)
        header["dtype"] = pickle.dumps(self.dtype)
        header["type-serialized"] = pickle.dumps(type(self))
        header["frame_count"] = len(frames)
        return header, frames

    def __contains__(self, item):
        return item in self._values

    @annotate("INDEX_EQUALS", color="green", domain="cudf_python")
    def equals(self, other, **kwargs):
        """
        Determine if two Index objects contain the same elements.

        Returns
        -------
        out: bool
            True if “other” is an Index and it has the same elements
            as calling index; False otherwise.
        """
        if not isinstance(other, BaseIndex):
            return False

        check_types = False

        self_is_categorical = isinstance(self, CategoricalIndex)
        other_is_categorical = isinstance(other, CategoricalIndex)
        if self_is_categorical and not other_is_categorical:
            other = other.astype(self.dtype)
            check_types = True
        elif other_is_categorical and not self_is_categorical:
            self = self.astype(other.dtype)
            check_types = True

        try:
            return super().equals(other, check_types=check_types)
        except TypeError:
            return False

    def get_level_values(self, level):
        """
        Return an Index of values for requested level.

        This is primarily useful to get an individual level of values from a
        MultiIndex, but is provided on Index as well for compatibility.

        Parameters
        ----------
        level : int or str
            It is either the integer position or the name of the level.

        Returns
        -------
        Index
            Calling object, as there is only one level in the Index.

        See Also
        --------
        cudf.core.multiindex.MultiIndex.get_level_values : Get values for
            a level of a MultiIndex.

        Notes
        -----
        For Index, level should be 0, since there are no multiple levels.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index(["a", "b", "c"])
        >>> idx.get_level_values(0)
        StringIndex(['a' 'b' 'c'], dtype='object')
        """

        if level == self.name:
            return self
        elif is_integer(level):
            if level != 0:
                raise IndexError(
                    f"Cannot get level: {level} " f"for index with 1 level"
                )
            return self
        else:
            raise KeyError(f"Requested level with name {level} " "not found")

    @classmethod
    def deserialize(cls, header, frames):
        h = header["index_column"]
        idx_typ = pickle.loads(header["type-serialized"])
        name = pickle.loads(header["name"])

        col_typ = pickle.loads(h["type-serialized"])
        index = col_typ.deserialize(h, frames[: header["frame_count"]])
        return idx_typ(index, name=name)

    @property
    def names(self):
        """
        Returns a tuple containing the name of the Index.
        """
        return (self.name,)

    @names.setter
    def names(self, values):
        if not is_list_like(values):
            raise ValueError("Names must be a list-like")

        num_values = len(values)
        if num_values > 1:
            raise ValueError(
                "Length of new names must be 1, got %d" % num_values
            )

        self.name = values[0]

    def dropna(self, how="any"):
        """
        Return an Index with null values removed.

        Parameters
        ----------
            how : {‘any’, ‘all’}, default ‘any’
                If the Index is a MultiIndex, drop the value when any or
                all levels are NaN.

        Returns
        -------
        valid : Index

        Examples
        --------
        >>> import cudf
        >>> index = cudf.Index(['a', None, 'b', 'c'])
        >>> index
        StringIndex(['a' None 'b' 'c'], dtype='object')
        >>> index.dropna()
        StringIndex(['a' 'b' 'c'], dtype='object')

        Using `dropna` on a `MultiIndex`:

        >>> midx = cudf.MultiIndex(
        ...         levels=[[1, None, 4, None], [1, 2, 5]],
        ...         codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        ...         names=["x", "y"],
        ...     )
        >>> midx
        MultiIndex([(   1, 1),
                    (   1, 5),
                    (<NA>, 2),
                    (   4, 2),
                    (<NA>, 1)],
                   names=['x', 'y'])
        >>> midx.dropna()
        MultiIndex([(1, 1),
                    (1, 5),
                    (4, 2)],
                   names=['x', 'y'])
        """
        return super().dropna(how=how)

    def _clean_nulls_from_index(self):
        """
        Convert all na values(if any) in Index object
        to `<NA>` as a preprocessing step to `__repr__` methods.

        This will involve changing type of Index object
        to StringIndex but it is the responsibility of the `__repr__`
        methods using this method to replace or handle representation
        of the actual types correctly.
        """
        if self._values.has_nulls:
            return cudf.Index(
                self._values.astype("str").fillna(cudf._NA_REP), name=self.name
            )
        else:
            return self

    @property
    def nlevels(self):
        """
        Number of levels.
        """
        return 1

    def _set_names(self, names, inplace=False):
        if inplace:
            idx = self
        else:
            idx = self.copy(deep=False)

        idx.names = names
        if not inplace:
            return idx

    def set_names(self, names, level=None, inplace=False):
        """
        Set Index or MultiIndex name.
        Able to set new names partially and by level.

        Parameters
        ----------
        names : label or list of label
            Name(s) to set.
        level : int, label or list of int or label, optional
            If the index is a MultiIndex, level(s) to set (None for all
            levels). Otherwise level must be None.
        inplace : bool, default False
            Modifies the object directly, instead of creating a new Index or
            MultiIndex.

        Returns
        -------
        Index
            The same type as the caller or None if inplace is True.

        See Also
        --------
        cudf.core.index.Index.rename : Able to set new names without level.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index([1, 2, 3, 4])
        >>> idx
        Int64Index([1, 2, 3, 4], dtype='int64')
        >>> idx.set_names('quarter')
        Int64Index([1, 2, 3, 4], dtype='int64', name='quarter')
        >>> idx = cudf.MultiIndex.from_product([['python', 'cobra'],
        ... [2018, 2019]])
        >>> idx
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   )
        >>> idx.names
        FrozenList([None, None])
        >>> idx.set_names(['kind', 'year'], inplace=True)
        >>> idx.names
        FrozenList(['kind', 'year'])
        >>> idx.set_names('species', level=0, inplace=True)
        >>> idx.names
        FrozenList(['species', 'year'])
        """
        if level is not None:
            raise ValueError("Level must be None for non-MultiIndex")

        if not is_list_like(names):
            names = [names]

        return self._set_names(names=names, inplace=inplace)

    def fillna(self, value, downcast=None):
        """
        Fill null values with the specified value.

        Parameters
        ----------
        value : scalar
            Scalar value to use to fill nulls. This value cannot be a
            list-likes.

        downcast : dict, default is None
            This Parameter is currently NON-FUNCTIONAL.

        Returns
        -------
        filled : Index

        Examples
        --------
        >>> import cudf
        >>> index = cudf.Index([1, 2, None, 4])
        >>> index
        Int64Index([1, 2, null, 4], dtype='int64')
        >>> index.fillna(3)
        Int64Index([1, 2, 3, 4], dtype='int64')
        """
        if downcast is not None:
            raise NotImplementedError(
                "`downcast` parameter is not yet supported"
            )

        return super().fillna(value=value)

    def take(self, indices):
        """Gather only the specific subset of indices

        Parameters
        ----------
        indices: An array-like that maps to values contained in this Index.
        """
        return self[indices]

    def argsort(self, ascending=True, **kwargs):
        """
        Return the integer indices that would sort the index.

        Parameters
        ----------
        ascending : bool, default True
            If True, returns the indices for ascending order.
            If False, returns the indices for descending order.

        Returns
        -------
        array : A cupy array containing Integer indices that
            would sort the index if used as an indexer.

        Examples
        --------
        >>> import cudf
        >>> index = cudf.Index([10, 100, 1, 1000])
        >>> index
        Int64Index([10, 100, 1, 1000], dtype='int64')
        >>> index.argsort()
        array([2, 0, 1, 3], dtype=int32)

        The order of argsort can be reversed using
        ``ascending`` parameter, by setting it to ``False``.
        >>> index.argsort(ascending=False)
        array([3, 1, 0, 2], dtype=int32)

        ``argsort`` on a MultiIndex:

        >>> index = cudf.MultiIndex(
        ...      levels=[[1, 3, 4, -10], [1, 11, 5]],
        ...      codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        ...      names=["x", "y"],
        ... )
        >>> index
        MultiIndex([(  1,  1),
                    (  1,  5),
                    (  3, 11),
                    (  4, 11),
                    (-10,  1)],
                   names=['x', 'y'])
        >>> index.argsort()
        array([4, 0, 1, 2, 3], dtype=int32)
        >>> index.argsort(ascending=False)
        array([3, 2, 1, 0, 4], dtype=int32)
        """
        indices = self._values.argsort(ascending=ascending, **kwargs)
        return cupy.asarray(indices)

    def to_frame(self, index=True, name=None):
        """Create a DataFrame with a column containing this Index

        Parameters
        ----------
        index : boolean, default True
            Set the index of the returned DataFrame as the original Index
        name : str, default None
            Name to be used for the column

        Returns
        -------
        DataFrame
            cudf DataFrame
        """

        if name is not None:
            col_name = name
        elif self.name is None:
            col_name = 0
        else:
            col_name = self.name
        return cudf.DataFrame(
            {col_name: self._values}, index=self if index else None
        )

    def any(self):
        """
        Return whether any elements is True in Index.
        """
        return self._values.any()

    def to_pandas(self):
        """
        Convert to a Pandas Index.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index([-3, 10, 15, 20])
        >>> idx
        Int64Index([-3, 10, 15, 20], dtype='int64')
        >>> idx.to_pandas()
        Int64Index([-3, 10, 15, 20], dtype='int64')
        >>> type(idx.to_pandas())
        <class 'pandas.core.indexes.numeric.Int64Index'>
        >>> type(idx)
        <class 'cudf.core.index.GenericIndex'>
        """
        return pd.Index(self._values.to_pandas(), name=self.name)

    @ioutils.doc_to_dlpack()
    def to_dlpack(self):
        """{docstring}"""

        return cudf.io.dlpack.to_dlpack(self)

    @property
    def gpu_values(self):
        """
        View the data as a numba device array object
        """
        return self._values.data_array_view

    def min(self):
        """
        Return the minimum value of the Index.

        Returns
        -------
        scalar
            Minimum value.

        See Also
        --------
        cudf.core.index.Index.max : Return the maximum value in an Index.
        cudf.core.series.Series.min : Return the minimum value in a Series.
        cudf.core.dataframe.DataFrame.min : Return the minimum values in
            a DataFrame.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index([3, 2, 1])
        >>> idx.min()
        1
        """
        return self._values.min()

    def max(self):
        """
        Return the maximum value of the Index.

        Returns
        -------
        scalar
            Maximum value.

        See Also
        --------
        cudf.core.index.Index.min : Return the minimum value in an Index.
        cudf.core.series.Series.max : Return the maximum value in a Series.
        cudf.core.dataframe.DataFrame.max : Return the maximum values in
            a DataFrame.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index([3, 2, 1])
        >>> idx.max()
        3
        """
        return self._values.max()

    def sum(self):
        """
        Return the sum of all values of the Index.

        Returns
        -------
        scalar
            Sum of all values.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index([3, 2, 1])
        >>> idx.sum()
        6
        """
        return self._values.sum()

    @classmethod
    def _concat(cls, objs):
        data = _concat_columns([o._values for o in objs])
        names = {obj.name for obj in objs}
        if len(names) == 1:
            [name] = names
        else:
            name = None
        result = as_index(data)
        result.name = name
        return result

    def append(self, other):
        """
        Append a collection of Index options together.

        Parameters
        ----------
        other : Index or list/tuple of indices

        Returns
        -------
        appended : Index

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index([1, 2, 10, 100])
        >>> idx
        Int64Index([1, 2, 10, 100], dtype='int64')
        >>> other = cudf.Index([200, 400, 50])
        >>> other
        Int64Index([200, 400, 50], dtype='int64')
        >>> idx.append(other)
        Int64Index([1, 2, 10, 100, 200, 400, 50], dtype='int64')

        append accepts list of Index objects

        >>> idx.append([other, other])
        Int64Index([1, 2, 10, 100, 200, 400, 50, 200, 400, 50], dtype='int64')
        """

        if is_list_like(other):
            to_concat = [self]
            to_concat.extend(other)
        else:
            this = self
            if len(other) == 0:
                # short-circuit and return a copy
                to_concat = [self]

            other = as_index(other)

            if len(self) == 0:
                to_concat = [other]

            if len(self) and len(other):
                if is_mixed_with_object_dtype(this, other):
                    got_dtype = (
                        other.dtype
                        if this.dtype == np.dtype("object")
                        else this.dtype
                    )
                    raise TypeError(
                        f"cudf does not support appending an Index of "
                        f"dtype `{np.dtype('object')}` with an Index "
                        f"of dtype `{got_dtype}`, please type-cast "
                        f"either one of them to same dtypes."
                    )

                if isinstance(self._values, cudf.core.column.NumericalColumn):
                    if self.dtype != other.dtype:
                        this, other = numeric_normalize_types(self, other)
                to_concat = [this, other]

        for obj in to_concat:
            if not isinstance(obj, BaseIndex):
                raise TypeError("all inputs must be Index")

        return self._concat(to_concat)

    def difference(self, other, sort=None):
        """
        Return a new Index with elements from the index that are not in
        `other`.

        This is the set difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        sort : False or None, default None
            Whether to sort the resulting index. By default, the
            values are attempted to be sorted, but any TypeError from
            incomparable elements is caught by cudf.

            * None : Attempt to sort the result, but catch any TypeErrors
              from comparing incomparable elements.
            * False : Do not sort the result.

        Returns
        -------
        difference : Index

        Examples
        --------
        >>> import cudf
        >>> idx1 = cudf.Index([2, 1, 3, 4])
        >>> idx1
        Int64Index([2, 1, 3, 4], dtype='int64')
        >>> idx2 = cudf.Index([3, 4, 5, 6])
        >>> idx2
        Int64Index([3, 4, 5, 6], dtype='int64')
        >>> idx1.difference(idx2)
        Int64Index([1, 2], dtype='int64')
        >>> idx1.difference(idx2, sort=False)
        Int64Index([2, 1], dtype='int64')
        """
        if sort not in {None, False}:
            raise ValueError(
                f"The 'sort' keyword only takes the values "
                f"of None or False; {sort} was passed."
            )

        other = as_index(other)

        if is_mixed_with_object_dtype(self, other):
            difference = self.copy()
        else:
            difference = self.join(other, how="leftanti")
            if self.dtype != other.dtype:
                difference = difference.astype(self.dtype)

        if sort is None:
            return difference.sort_values()

        return difference

    def _binaryop(self, other, fn, fill_value=None, reflect=False):
        # TODO: Rather than including an allowlist of acceptable types, we
        # should instead return NotImplemented for __all__ other types. That
        # will allow other types to support binops with cudf objects if they so
        # choose, and just as importantly will allow better error messages if
        # they don't support it.
        if isinstance(other, (cudf.DataFrame, cudf.Series)):
            return NotImplemented

        return super()._binaryop(other, fn, fill_value, reflect)

    def _copy_construct(self, **kwargs):
        # Need to override the parent behavior because pandas allows operations
        # on unsigned types to return signed values, forcing us to choose the
        # right index type here.
        data = kwargs.get("data")
        cls = self.__class__

        if data is not None:
            if self.dtype != data.dtype:
                # TODO: This logic is largely copied from `as_index`. The two
                # should be unified via a centralized type dispatching scheme.
                if isinstance(data, NumericalColumn):
                    try:
                        cls = _dtype_to_index[data.dtype.type]
                    except KeyError:
                        cls = GenericIndex
                elif isinstance(data, StringColumn):
                    cls = StringIndex
                elif isinstance(data, DatetimeColumn):
                    cls = DatetimeIndex
                elif isinstance(data, TimeDeltaColumn):
                    cls = TimedeltaIndex
                elif isinstance(data, CategoricalColumn):
                    cls = CategoricalIndex
            elif cls is RangeIndex:
                # RangeIndex must convert to other numerical types for ops

                # TODO: The one exception to the output type selected here is
                # that scalar multiplication of a RangeIndex in pandas results
                # in another RangeIndex. Propagating that information through
                # cudf with the current internals is possible, but requires
                # significant hackery since we'd need _copy_construct or some
                # other constructor to be intrinsically capable of processing
                # operations. We should fix this behavior once we've completed
                # a more thorough refactoring of the various Index classes that
                # makes it easier to propagate this logic.
                try:
                    cls = _dtype_to_index[data.dtype.type]
                except KeyError:
                    cls = GenericIndex

        return cls(**{**self._copy_construct_defaults, **kwargs})

    def sort_values(self, return_indexer=False, ascending=True, key=None):
        """
        Return a sorted copy of the index, and optionally return the indices
        that sorted the index itself.

        Parameters
        ----------
        return_indexer : bool, default False
            Should the indices that would sort the index be returned.
        ascending : bool, default True
            Should the index values be sorted in an ascending order.
        key : None, optional
            This parameter is NON-FUNCTIONAL.

        Returns
        -------
        sorted_index : Index
            Sorted copy of the index.
        indexer : cupy.ndarray, optional
            The indices that the index itself was sorted by.

        See Also
        --------
        cudf.core.series.Series.min : Sort values of a Series.
        cudf.core.dataframe.DataFrame.sort_values : Sort values in a DataFrame.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index([10, 100, 1, 1000])
        >>> idx
        Int64Index([10, 100, 1, 1000], dtype='int64')

        Sort values in ascending order (default behavior).

        >>> idx.sort_values()
        Int64Index([1, 10, 100, 1000], dtype='int64')

        Sort values in descending order, and also get the indices `idx` was
        sorted by.

        >>> idx.sort_values(ascending=False, return_indexer=True)
        (Int64Index([1000, 100, 10, 1], dtype='int64'), array([3, 1, 0, 2],
                                                            dtype=int32))

        Sorting values in a MultiIndex:

        >>> midx = cudf.MultiIndex(
        ...      levels=[[1, 3, 4, -10], [1, 11, 5]],
        ...      codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        ...      names=["x", "y"],
        ... )
        >>> midx
        MultiIndex([(  1,  1),
                    (  1,  5),
                    (  3, 11),
                    (  4, 11),
                    (-10,  1)],
                   names=['x', 'y'])
        >>> midx.sort_values()
        MultiIndex([(-10,  1),
                    (  1,  1),
                    (  1,  5),
                    (  3, 11),
                    (  4, 11)],
                   names=['x', 'y'])
        >>> midx.sort_values(ascending=False)
        MultiIndex([(  4, 11),
                    (  3, 11),
                    (  1,  5),
                    (  1,  1),
                    (-10,  1)],
                   names=['x', 'y'])
        """
        if key is not None:
            raise NotImplementedError("key parameter is not yet implemented.")

        indices = self._values.argsort(ascending=ascending)
        index_sorted = as_index(self.take(indices), name=self.name)

        if return_indexer:
            return index_sorted, cupy.asarray(indices)
        else:
            return index_sorted

    def unique(self):
        """
        Return unique values in the index.

        Returns
        -------
        Index without duplicates
        """
        return as_index(self._values.unique(), name=self.name)

    def join(
        self, other, how="left", level=None, return_indexers=False, sort=False
    ):
        """
        Compute join_index and indexers to conform data structures
        to the new index.

        Parameters
        ----------
        other : Index.
        how : {'left', 'right', 'inner', 'outer'}
        return_indexers : bool, default False
        sort : bool, default False
            Sort the join keys lexicographically in the result Index. If False,
            the order of the join keys depends on the join type (how keyword).

        Returns: index

        Examples
        --------
        >>> import cudf
        >>> lhs = cudf.DataFrame(
        ...     {"a":[2, 3, 1], "b":[3, 4, 2]}).set_index(['a', 'b']
        ... ).index
        >>> lhs
        MultiIndex([(2, 3),
                    (3, 4),
                    (1, 2)],
                   names=['a', 'b'])
        >>> rhs = cudf.DataFrame({"a":[1, 4, 3]}).set_index('a').index
        >>> rhs
        Int64Index([1, 4, 3], dtype='int64', name='a')
        >>> lhs.join(rhs, how='inner')
        MultiIndex([(3, 4),
                    (1, 2)],
                   names=['a', 'b'])
        """

        if isinstance(self, cudf.MultiIndex) and isinstance(
            other, cudf.MultiIndex
        ):
            raise TypeError(
                "Join on level between two MultiIndex objects is ambiguous"
            )

        if level is not None and not is_scalar(level):
            raise ValueError("level should be an int or a label only")

        if isinstance(other, cudf.MultiIndex):
            if how == "left":
                how = "right"
            elif how == "right":
                how = "left"
            rhs = self.copy(deep=False)
            lhs = other.copy(deep=False)
        else:
            lhs = self.copy(deep=False)
            rhs = other.copy(deep=False)

        on = level
        # In case of MultiIndex, it will be None as
        # we don't need to update name
        left_names = lhs.names
        right_names = rhs.names
        # There should be no `None` values in Joined indices,
        # so essentially it would be `left/right` or 'inner'
        # in case of MultiIndex
        if isinstance(lhs, cudf.MultiIndex):
            if level is not None and isinstance(level, int):
                on = lhs._data.select_by_index(level).names[0]
            right_names = (on,) or right_names
            on = right_names[0]
            if how == "outer":
                how = "left"
            elif how == "right":
                how = "inner"
        else:
            # Both are nomal indices
            right_names = left_names
            on = right_names[0]

        lhs.names = left_names
        rhs.names = right_names

        output = lhs._merge(rhs, how=how, on=on, sort=sort)

        return output

    def rename(self, name, inplace=False):
        """
        Alter Index name.

        Defaults to returning new index.

        Parameters
        ----------
        name : label
            Name(s) to set.

        Returns
        -------
        Index

        Examples
        --------
        >>> import cudf
        >>> index = cudf.Index([1, 2, 3], name='one')
        >>> index
        Int64Index([1, 2, 3], dtype='int64', name='one')
        >>> index.name
        'one'
        >>> renamed_index = index.rename('two')
        >>> renamed_index
        Int64Index([1, 2, 3], dtype='int64', name='two')
        >>> renamed_index.name
        'two'
        """
        if inplace is True:
            self.name = name
            return None
        else:
            out = self.copy(deep=False)
            out.name = name
            return out.copy(deep=True)

    def astype(self, dtype, copy=False):
        """
        Create an Index with values cast to dtypes. The class of a new Index
        is determined by dtype. When conversion is impossible, a ValueError
        exception is raised.

        Parameters
        ----------
        dtype : numpy dtype
            Use a numpy.dtype to cast entire Index object to.
        copy : bool, default False
            By default, astype always returns a newly allocated object.
            If copy is set to False and internal requirements on dtype are
            satisfied, the original data is used to create a new Index
            or the original Index is returned.

        Returns
        -------
        Index
            Index with values cast to specified dtype.

        Examples
        --------
        >>> import cudf
        >>> index = cudf.Index([1, 2, 3])
        >>> index
        Int64Index([1, 2, 3], dtype='int64')
        >>> index.astype('float64')
        Float64Index([1.0, 2.0, 3.0], dtype='float64')
        """
        if is_dtype_equal(dtype, self.dtype):
            return self.copy(deep=copy)

        return as_index(
            self.copy(deep=copy)._values.astype(dtype), name=self.name
        )

    def to_array(self, fillna=None):
        """Get a dense numpy array for the data.

        Parameters
        ----------
        fillna : str or None
            Defaults to None, which will skip null values.
            If it equals "pandas", null values are filled with NaNs.
            Non integral dtype is promoted to np.float64.

        Notes
        -----

        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.
        """
        return self._values.to_array(fillna=fillna)

    def to_series(self, index=None, name=None):
        """
        Create a Series with both index and values equal to the index keys.
        Useful with map for returning an indexer based on an index.

        Parameters
        ----------
        index : Index, optional
            Index of resulting Series. If None, defaults to original index.
        name : str, optional
            Dame of resulting Series. If None, defaults to name of original
            index.

        Returns
        -------
        Series
            The dtype will be based on the type of the Index values.
        """
        return cudf.Series(
            self._values,
            index=self.copy(deep=False) if index is None else index,
            name=self.name if name is None else name,
        )

    def get_slice_bound(self, label, side, kind):
        """
        Calculate slice bound that corresponds to given label.
        Returns leftmost (one-past-the-rightmost if ``side=='right'``) position
        of given label.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}
        kind : {'ix', 'loc', 'getitem'}

        Returns
        -------
        int
            Index of label.
        """
        raise (NotImplementedError)

    def __array_function__(self, func, types, args, kwargs):

        # check if the function is implemented for the current type
        cudf_index_module = type(self)
        for submodule in func.__module__.split(".")[1:]:
            # point cudf_index_module to the correct submodule
            if hasattr(cudf_index_module, submodule):
                cudf_index_module = getattr(cudf_index_module, submodule)
            else:
                return NotImplemented

        fname = func.__name__

        handled_types = [Index, cudf.Series]

        # check if  we don't handle any of the types (including sub-class)
        for t in types:
            if not any(
                issubclass(t, handled_type) for handled_type in handled_types
            ):
                return NotImplemented

        if hasattr(cudf_index_module, fname):
            cudf_func = getattr(cudf_index_module, fname)
            # Handle case if cudf_func is same as numpy function
            if cudf_func is func:
                return NotImplemented
            else:
                return cudf_func(*args, **kwargs)

        else:
            return NotImplemented

    def isin(self, values):
        """Return a boolean array where the index values are in values.

        Compute boolean array of whether each index value is found in
        the passed set of values. The length of the returned boolean
        array matches the length of the index.

        Parameters
        ----------
        values : set, list-like, Index
            Sought values.

        Returns
        -------
        is_contained : cupy array
            CuPy array of boolean values.

        Examples
        --------
        >>> idx = cudf.Index([1,2,3])
        >>> idx
        Int64Index([1, 2, 3], dtype='int64')

        Check whether each index value in a list of values.

        >>> idx.isin([1, 4])
        array([ True, False, False])
        """

        return self._values.isin(values).values

    def where(self, cond, other=None):
        """
        Replace values where the condition is False.

        Parameters
        ----------
        cond : bool array-like with the same length as self
            Where cond is True, keep the original value.
            Where False, replace with corresponding value from other.
            Callables are not supported.
        other: scalar, or array-like
            Entries where cond is False are replaced with
            corresponding value from other. Callables are not
            supported. Default is None.

        Returns
        -------
        Same type as caller

        Examples
        --------
        >>> import cudf
        >>> index = cudf.Index([4, 3, 2, 1, 0])
        >>> index
        Int64Index([4, 3, 2, 1, 0], dtype='int64')
        >>> index.where(index > 2, 15)
        Int64Index([4, 3, 15, 15, 15], dtype='int64')
        """
        return super().where(cond=cond, other=other)

    def memory_usage(self, deep=False):
        """
        Memory usage of the values.

        Parameters
        ----------
            deep : bool
                Introspect the data deeply,
                interrogate `object` dtypes for system-level
                memory consumption.

        Returns
        -------
            bytes used
        """
        return self._values._memory_usage(deep=deep)

    def get_loc(self, key, method=None, tolerance=None):
        """Get integer location, slice or boolean mask for requested label.

        Parameters
        ----------
        key : label
        method : {None, 'pad'/'fill', 'backfill'/'bfill', 'nearest'}, optional
            - default: exact matches only.
            - pad / ffill: find the PREVIOUS index value if no exact match.
            - backfill / bfill: use NEXT index value if no exact match.
            - nearest: use the NEAREST index value if no exact match. Tied
              distances are broken by preferring the larger index
              value.
        tolerance : int or float, optional
            Maximum distance from index value for inexact matches. The value
            of the index at the matching location must satisfy the equation
            ``abs(index[loc] - key) <= tolerance``.

        Returns
        -------
        int or slice or boolean mask
            - If result is unique, return integer index
            - If index is monotonic, loc is returned as a slice object
            - Otherwise, a boolean mask is returned

        Examples
        --------
        >>> unique_index = cudf.Index(list('abc'))
        >>> unique_index.get_loc('b')
        1
        >>> monotonic_index = cudf.Index(list('abbc'))
        >>> monotonic_index.get_loc('b')
        slice(1, 3, None)
        >>> non_monotonic_index = cudf.Index(list('abcb'))
        >>> non_monotonic_index.get_loc('b')
        array([False,  True, False,  True])
        >>> numeric_unique_index = cudf.Index([1, 2, 3])
        >>> numeric_unique_index.get_loc(3)
        2
        """
        if tolerance is not None:
            raise NotImplementedError(
                "Parameter tolerance is unsupported yet."
            )
        if method not in {
            None,
            "ffill",
            "bfill",
            "pad",
            "backfill",
            "nearest",
        }:
            raise ValueError(
                f"Invalid fill method. Expecting pad (ffill), backfill (bfill)"
                f" or nearest. Got {method}"
            )

        is_sorted = (
            self.is_monotonic_increasing or self.is_monotonic_decreasing
        )

        if not is_sorted and method is not None:
            raise ValueError(
                "index must be monotonic increasing or decreasing if `method`"
                "is specified."
            )

        key_as_table = Table({"None": as_column(key, length=1)})
        lower_bound, upper_bound, sort_inds = self._lexsorted_equal_range(
            key_as_table, is_sorted
        )

        if lower_bound == upper_bound:
            # Key not found, apply method
            if method in ("pad", "ffill"):
                if lower_bound == 0:
                    raise KeyError(key)
                return lower_bound - 1
            elif method in ("backfill", "bfill"):
                if lower_bound == self._data.nrows:
                    raise KeyError(key)
                return lower_bound
            elif method == "nearest":
                if lower_bound == self._data.nrows:
                    return lower_bound - 1
                elif lower_bound == 0:
                    return 0
                lower_val = self._column.element_indexing(lower_bound - 1)
                upper_val = self._column.element_indexing(lower_bound)
                return (
                    lower_bound - 1
                    if abs(lower_val - key) < abs(upper_val - key)
                    else lower_bound
                )
            else:
                raise KeyError(key)

        if lower_bound + 1 == upper_bound:
            # Search result is unique, return int.
            return (
                lower_bound
                if is_sorted
                else sort_inds.element_indexing(lower_bound)
            )

        if is_sorted:
            # In monotonic index, lex search result is continuous. A slice for
            # the range is returned.
            return slice(lower_bound, upper_bound)

        # Not sorted and not unique. Return a boolean mask
        mask = cupy.full(self._data.nrows, False)
        true_inds = sort_inds.slice(lower_bound, upper_bound).to_gpu_array()
        mask[cupy.array(true_inds)] = True
        return mask

    def _lexsorted_equal_range(
        self, key_as_table: Table, is_sorted: bool
    ) -> Tuple[int, int, Optional[ColumnBase]]:
        """Get equal range for key in lexicographically sorted index. If index
        is not sorted when called, a sort will take place and `sort_inds` is
        returned. Otherwise `None` is returned in that position.
        """
        if not is_sorted:
            sort_inds = self._get_sorted_inds()
            sort_vals = self._gather(sort_inds)
        else:
            sort_inds = None
            sort_vals = self
        lower_bound = search_sorted(
            sort_vals, key_as_table, side="left"
        ).element_indexing(0)
        upper_bound = search_sorted(
            sort_vals, key_as_table, side="right"
        ).element_indexing(0)

        return lower_bound, upper_bound, sort_inds

    @classmethod
    def from_pandas(cls, index, nan_as_null=None):
        """
        Convert from a Pandas Index.

        Parameters
        ----------
        index : Pandas Index object
            A Pandas Index object which has to be converted
            to cuDF Index.
        nan_as_null : bool, Default None
            If ``None``/``True``, converts ``np.nan`` values
            to ``null`` values.
            If ``False``, leaves ``np.nan`` values as is.

        Raises
        ------
        TypeError for invalid input type.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = [10, 20, 30, np.nan]
        >>> pdi = pd.Index(data)
        >>> cudf.core.index.Index.from_pandas(pdi)
        Float64Index([10.0, 20.0, 30.0, <NA>], dtype='float64')
        >>> cudf.core.index.Index.from_pandas(pdi, nan_as_null=False)
        Float64Index([10.0, 20.0, 30.0, nan], dtype='float64')
        """
        if not isinstance(index, pd.Index):
            raise TypeError("not a pandas.Index")

        ind = as_index(column.as_column(index, nan_as_null=nan_as_null))
        ind.name = index.name
        return ind

    @classmethod
    def _from_table(cls, table):
        if not isinstance(table, RangeIndex):
            if table._num_columns == 0:
                raise ValueError("Cannot construct Index from any empty Table")
            if table._num_columns == 1:
                values = next(iter(table._data.values()))

                if isinstance(values, NumericalColumn):
                    try:
                        index_class_type = _dtype_to_index[values.dtype.type]
                    except KeyError:
                        index_class_type = GenericIndex
                    out = super(BaseIndex, index_class_type).__new__(
                        index_class_type
                    )
                elif isinstance(values, DatetimeColumn):
                    out = super(BaseIndex, DatetimeIndex).__new__(
                        DatetimeIndex
                    )
                elif isinstance(values, TimeDeltaColumn):
                    out = super(BaseIndex, TimedeltaIndex).__new__(
                        TimedeltaIndex
                    )
                elif isinstance(values, StringColumn):
                    out = super(BaseIndex, StringIndex).__new__(StringIndex)
                elif isinstance(values, CategoricalColumn):
                    out = super(BaseIndex, CategoricalIndex).__new__(
                        CategoricalIndex
                    )
                out._data = table._data
                out._index = None
                return out
            else:
                return cudf.MultiIndex._from_table(
                    table, names=table._data.names
                )
        else:
            return as_index(table)

    @property
    def _copy_construct_defaults(self):
        return {"data": self._column, "name": self.name}

    @classmethod
    def _from_data(cls, data, index=None):
        return cls._from_table(SingleColumnFrame(data=data))

    @property
    def _constructor_expanddim(self):
        return cudf.MultiIndex


class RangeIndex(BaseIndex):
    """
    Immutable Index implementing a monotonic integer range.

    This is the default index type used by DataFrame and Series
    when no explicit index is provided by the user.

    Parameters
    ----------
    start : int (default: 0), or other range instance
    stop : int (default: 0)
    step : int (default: 1)
    name : object, optional
        Name to be stored in the index.
    dtype : numpy dtype
        Unused, accepted for homogeneity with other index types.
    copy : bool, default False
        Unused, accepted for homogeneity with other index types.

    Returns
    -------
    RangeIndex

    Examples
    --------
    >>> import cudf
    >>> cudf.RangeIndex(0, 10, 1, name="a")
    RangeIndex(start=0, stop=10, step=1, name='a')

    >>> cudf.RangeIndex(range(1, 10, 1), name="a")
    RangeIndex(start=1, stop=10, step=1, name='a')
    """

    def __init__(
        self, start, stop=None, step=1, dtype=None, copy=False, name=None
    ):
        if step == 0:
            raise ValueError("Step must not be zero.")

        if isinstance(start, range):
            therange = start
            start = therange.start
            stop = therange.stop
            step = therange.step
        if stop is None:
            start, stop = 0, start
        self._start = int(start)
        self._stop = int(stop)
        self._step = int(step) if step is not None else 1
        self._index = None
        self._name = name

    @property
    def name(self):
        """
        Returns the name of the Index.
        """
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def start(self):
        """
        The value of the `start` parameter (0 if this was not supplied).
        """
        return self._start

    @property
    def stop(self):
        """
        The value of the stop parameter.
        """
        return self._stop

    @property
    def step(self):
        """
        The value of the step parameter.
        """
        return self._step

    @property
    def _num_rows(self):
        return len(self)

    @cached_property
    def _values(self):
        if len(self) > 0:
            return column.arange(
                self._start, self._stop, self._step, dtype=self.dtype
            )
        else:
            return column.column_empty(0, masked=False, dtype=self.dtype)

    @property
    def _data(self):
        return cudf.core.column_accessor.ColumnAccessor(
            {self.name: self._values}
        )

    def __contains__(self, item):
        if not isinstance(
            item, tuple(np.sctypes["int"] + np.sctypes["float"] + [int, float])
        ):
            return False
        if not item % 1 == 0:
            return False
        return item in range(self._start, self._stop, self._step)

    def copy(self, name=None, deep=False, dtype=None, names=None):
        """
        Make a copy of this object.

        Parameters
        ----------
        name : object optional (default: None), name of index
        deep : Bool (default: False)
            Ignored for RangeIndex
        dtype : numpy dtype optional (default: None)
            Target dtype for underlying range data
        names : list-like optional (default: False)
            Kept compatibility with MultiIndex. Should not be used.

        Returns
        -------
        New RangeIndex instance with same range, casted to new dtype
        """

        dtype = self.dtype if dtype is None else dtype

        if not np.issubdtype(dtype, np.signedinteger):
            raise ValueError(f"Expected Signed Integer Type, Got {dtype}")

        name = self.name if name is None else name

        return RangeIndex(
            start=self._start, stop=self._stop, step=self._step, name=name
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(start={self._start}, stop={self._stop}"
            f", step={self._step}"
            + (
                f", name={pd.io.formats.printing.default_pprint(self.name)}"
                if self.name is not None
                else ""
            )
            + ")"
        )

    def __len__(self):
        return len(range(self._start, self._stop, self._step))

    def __getitem__(self, index):
        len_self = len(self)
        if isinstance(index, slice):
            sl_start, sl_stop, sl_step = index.indices(len_self)

            lo = self._start + sl_start * self._step
            hi = self._start + sl_stop * self._step
            st = self._step * sl_step
            return RangeIndex(start=lo, stop=hi, step=st, name=self._name)

        elif isinstance(index, Number):
            if index < 0:
                index = len_self + index
            if not (0 <= index < len_self):
                raise IndexError("out-of-bound")
            index = min(index, len_self)
            index = self._start + index * self._step
            return index
        else:
            if _is_scalar_or_zero_d_array(index):
                index = np.min_scalar_type(index).type(index)
            index = column.as_column(index)

        return as_index(self._values[index], name=self.name)

    def equals(self, other):
        if isinstance(other, RangeIndex):
            if (self._start, self._stop, self._step) == (
                other._start,
                other._stop,
                other._step,
            ):
                return True
        return super().equals(other)

    def serialize(self):
        header = {}
        header["index_column"] = {}

        # store metadata values of index separately
        # We don't need to store the GPU buffer for RangeIndexes
        # cuDF only needs to store start/stop and rehydrate
        # during de-serialization
        header["index_column"]["start"] = self._start
        header["index_column"]["stop"] = self._stop
        header["index_column"]["step"] = self._step
        frames = []

        header["name"] = pickle.dumps(self.name)
        header["dtype"] = pickle.dumps(self.dtype)
        header["type-serialized"] = pickle.dumps(type(self))
        header["frame_count"] = 0
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        h = header["index_column"]
        name = pickle.loads(header["name"])
        start = h["start"]
        stop = h["stop"]
        step = h.get("step", 1)
        return RangeIndex(start=start, stop=stop, step=step, name=name)

    @property
    def dtype(self):
        """
        `dtype` of the range of values in RangeIndex.
        """
        return np.dtype(np.int64)

    @property
    def is_contiguous(self):
        """
        Returns if the index is contiguous.
        """
        return self._step == 1

    @property
    def size(self):
        return len(self)

    def find_label_range(self, first=None, last=None):
        """Find subrange in the ``RangeIndex``, marked by their positions, that
        starts greater or equal to ``first`` and ends less or equal to ``last``

        The range returned is assumed to be monotonically increasing. In cases
        where there is no such range that suffice the constraint, an exception
        will be raised.

        Parameters
        ----------
        first, last : int, optional, Default None
            The "start" and "stop" values of the subrange. If None, will use
            ``self._start`` as first, ``self._stop`` as last.

        Returns
        -------
        begin, end : 2-tuple of int
            The starting index and the ending index.
            The `last` value occurs at ``end - 1`` position.
        """

        first = self._start if first is None else first
        last = self._stop if last is None else last

        if self._step < 0:
            first = -first
            last = -last
            start = -self._start
            step = -self._step
        else:
            start = self._start
            step = self._step

        stop = start + len(self) * step
        begin = search_range(start, stop, first, step, side="left")
        end = search_range(start, stop, last, step, side="right")

        return begin, end

    def to_pandas(self):
        return pd.RangeIndex(
            start=self._start,
            stop=self._stop,
            step=self._step,
            dtype=self.dtype,
            name=self.name,
        )

    @property
    def is_unique(self):
        """
        Return if the index has unique values.
        """
        return True

    @property
    def is_monotonic_increasing(self):
        """
        Return if the index is monotonic increasing
        (only equal or increasing) values.
        """
        return self._step > 0 or len(self) <= 1

    @property
    def is_monotonic_decreasing(self):
        """
        Return if the index is monotonic decreasing
        (only equal or decreasing) values.
        """
        return self._step < 0 or len(self) <= 1

    def get_slice_bound(self, label, side, kind=None):
        """
        Calculate slice bound that corresponds to given label.
        Returns leftmost (one-past-the-rightmost if ``side=='right'``) position
        of given label.

        Parameters
        ----------
        label : int
            A valid value in the ``RangeIndex``
        side : {'left', 'right'}
        kind : Unused
            To keep consistency with other index types.

        Returns
        -------
        int
            Index of label.
        """
        if side not in {"left", "right"}:
            raise ValueError(f"Unrecognized side parameter: {side}")

        if self._step < 0:
            label = -label
            start = -self._start
            step = -self._step
        else:
            start = self._start
            step = self._step

        stop = start + len(self) * step
        pos = search_range(start, stop, label, step, side=side)
        return pos

    def memory_usage(self, **kwargs):
        return 0

    def unique(self):
        # RangeIndex always has unique values
        return self


class GenericIndex(BaseIndex):
    """An array of orderable values that represent the indices of another Column

    Attributes
    ----------
    _values: A Column object
    name: A string
    """

    def __init__(self, data, **kwargs):
        """
        Parameters
        ----------
        data : Column
            The Column of data for this index
        name : str optional
            The name of the Index. If not provided, the Index adopts the value
            Column's name. Otherwise if this name is different from the value
            Column's, the data Column will be cloned to adopt this name.
        """
        kwargs = _setdefault_name(data, **kwargs)

        # normalize the input
        if isinstance(data, cudf.Series):
            data = data._column
        elif isinstance(data, column.ColumnBase):
            data = data
        else:
            if isinstance(data, (list, tuple)):
                if len(data) == 0:
                    data = np.asarray([], dtype="int64")
                else:
                    data = np.asarray(data)
            data = column.as_column(data)
            assert isinstance(data, (NumericalColumn, StringColumn))

        name = kwargs.get("name")
        super().__init__({name: data})

    @property
    def _values(self):
        return self._column

    def copy(self, name=None, deep=False, dtype=None, names=None):
        """
        Make a copy of this object.

        Parameters
        ----------
        name : object, default None
            Name of index, use original name when None
        deep : bool, default True
            Make a deep copy of the data.
            With ``deep=False`` the original data is used
        dtype : numpy dtype, default None
            Target datatype to cast into, use original dtype when None
        names : list-like, default False
            Kept compatibility with MultiIndex. Should not be used.

        Returns
        -------
        New index instance, casted to new dtype
        """

        dtype = self.dtype if dtype is None else dtype
        name = self.name if name is None else name

        return as_index(self._values.astype(dtype), name=name, copy=deep)

    def __sizeof__(self):
        return self._values.__sizeof__()

    def __repr__(self):
        max_seq_items = get_option("max_seq_items") or len(self)
        mr = 0
        if 2 * max_seq_items < len(self):
            mr = max_seq_items + 1

        if len(self) > mr and mr != 0:
            top = self[0:mr]
            bottom = self[-1 * mr :]

            preprocess = cudf.concat([top, bottom])
        else:
            preprocess = self

        # TODO: Change below usages accordingly to
        # utilize `Index.to_string` once it is implemented
        # related issue : https://github.com/pandas-dev/pandas/issues/35389
        if isinstance(preprocess, CategoricalIndex):
            if preprocess.categories.dtype.kind == "f":
                output = (
                    preprocess.astype("str")
                    .to_pandas()
                    .astype(
                        dtype=pd.CategoricalDtype(
                            categories=preprocess.dtype.categories.astype(
                                "str"
                            ).to_pandas(),
                            ordered=preprocess.dtype.ordered,
                        )
                    )
                    .__repr__()
                )
                break_idx = output.find("ordered=")
                output = (
                    output[:break_idx].replace("'", "") + output[break_idx:]
                )
            else:
                output = preprocess.to_pandas().__repr__()

            output = output.replace("nan", cudf._NA_REP)
        elif preprocess._values.nullable:
            output = self._clean_nulls_from_index().to_pandas().__repr__()

            if not isinstance(self, StringIndex):
                # We should remove all the single quotes
                # from the output due to the type-cast to
                # object dtype happening above.
                # Note : The replacing of single quotes has
                # to happen only incase of non-StringIndex types,
                # as we want to preserve single quotes incase
                # of StringIndex and it is valid to have them.
                output = output.replace("'", "")
        else:
            output = preprocess.to_pandas().__repr__()

        # Fix and correct the class name of the output
        # string by finding first occurrence of "(" in the output
        index_class_split_index = output.find("(")
        output = self.__class__.__name__ + output[index_class_split_index:]

        lines = output.split("\n")

        tmp_meta = lines[-1]
        dtype_index = tmp_meta.rfind(" dtype=")
        prior_to_dtype = tmp_meta[:dtype_index]
        lines = lines[:-1]
        lines.append(prior_to_dtype + " dtype='%s'" % self.dtype)
        if self.name is not None:
            lines[-1] = lines[-1] + ", name='%s'" % self.name
        if "length" in tmp_meta:
            lines[-1] = lines[-1] + ", length=%d)" % len(self)
        else:
            lines[-1] = lines[-1] + ")"

        return "\n".join(lines)

    def __getitem__(self, index):
        if type(self) == IntervalIndex:
            raise NotImplementedError(
                "Getting a scalar from an IntervalIndex is not yet supported"
            )
        res = self._values[index]
        if not isinstance(index, int):
            res = as_index(res)
            res.name = self.name
        return res

    @property
    def dtype(self):
        """
        `dtype` of the underlying values in GenericIndex.
        """
        return self._values.dtype

    def find_label_range(self, first, last):
        """Find range that starts with *first* and ends with *last*,
        inclusively.

        Returns
        -------
        begin, end : 2-tuple of int
            The starting index and the ending index.
            The *last* value occurs at ``end - 1`` position.
        """
        col = self._values
        begin, end = None, None
        if first is not None:
            begin = col.find_first_value(first, closest=True)
        if last is not None:
            end = col.find_last_value(last, closest=True)
            end += 1
        return begin, end

    def get_slice_bound(self, label, side, kind):
        return self._values.get_slice_bound(label, side, kind)


class NumericIndex(GenericIndex):
    """Immutable, ordered and sliceable sequence of labels.
    The basic object storing row labels for all cuDF objects.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : NumPy dtype,
            but not used.
    copy : bool
        Make a copy of input data.
    name : object
        Name to be stored in the index.

    Returns
    -------
    Index
    """

    # Subclasses must define the dtype they are associated with.
    _dtype: Union[None, Type[np.number]] = None

    def __init__(self, data=None, dtype=None, copy=False, name=None):

        dtype = type(self)._dtype
        if copy:
            data = column.as_column(data, dtype=dtype).copy()

        kwargs = _setdefault_name(data, name=name)

        data = column.as_column(data, dtype=dtype)

        super().__init__(data, **kwargs)


class Int8Index(NumericIndex):
    _dtype = np.int8


class Int16Index(NumericIndex):
    _dtype = np.int16


class Int32Index(NumericIndex):
    _dtype = np.int32


class Int64Index(NumericIndex):
    _dtype = np.int64


class UInt8Index(NumericIndex):
    _dtype = np.uint8


class UInt16Index(NumericIndex):
    _dtype = np.uint16


class UInt32Index(NumericIndex):
    _dtype = np.uint32


class UInt64Index(NumericIndex):
    _dtype = np.uint64


class Float32Index(NumericIndex):
    _dtype = np.float32


class Float64Index(NumericIndex):
    _dtype = np.float64


class DatetimeIndex(GenericIndex):
    """
    Immutable , ordered and sliceable sequence of datetime64 data,
    represented internally as int64.

    Parameters
    ----------
    data : array-like (1-dimensional), optional
        Optional datetime-like data to construct index with.
    copy : bool
        Make a copy of input.
    freq : str, optional
        This is not yet supported
    tz : pytz.timezone or dateutil.tz.tzfile
        This is not yet supported
    ambiguous : ‘infer’, bool-ndarray, ‘NaT’, default ‘raise’
        This is not yet supported
    name : object
        Name to be stored in the index.
    dayfirst : bool, default False
        If True, parse dates in data with the day first order.
        This is not yet supported
    yearfirst : bool, default False
        If True parse dates in data with the year first order.
        This is not yet supported

    Returns
    -------
    DatetimeIndex

    Examples
    --------
    >>> import cudf
    >>> cudf.DatetimeIndex([1, 2, 3, 4], name="a")
    DatetimeIndex(['1970-01-01 00:00:00.001000', '1970-01-01 00:00:00.002000',
                   '1970-01-01 00:00:00.003000', '1970-01-01 00:00:00.004000'],
                  dtype='datetime64[ms]', name='a')
    """

    def __init__(
        self,
        data=None,
        freq=None,
        tz=None,
        normalize=False,
        closed=None,
        ambiguous="raise",
        dayfirst=False,
        yearfirst=False,
        dtype=None,
        copy=False,
        name=None,
    ):
        # we should be more strict on what we accept here but
        # we'd have to go and figure out all the semantics around
        # pandas dtindex creation first which.  For now
        # just make sure we handle np.datetime64 arrays
        # and then just dispatch upstream
        if freq is not None:
            raise NotImplementedError("Freq is not yet supported")
        if tz is not None:
            raise NotImplementedError("tz is not yet supported")
        if normalize is not False:
            raise NotImplementedError("normalize == True is not yet supported")
        if closed is not None:
            raise NotImplementedError("closed is not yet supported")
        if ambiguous != "raise":
            raise NotImplementedError("ambiguous is not yet supported")
        if dayfirst is not False:
            raise NotImplementedError("dayfirst == True is not yet supported")
        if yearfirst is not False:
            raise NotImplementedError("yearfirst == True is not yet supported")

        valid_dtypes = tuple(
            f"datetime64[{res}]" for res in ("s", "ms", "us", "ns")
        )
        if dtype is None:
            # nanosecond default matches pandas
            dtype = "datetime64[ns]"
        elif dtype not in valid_dtypes:
            raise TypeError("Invalid dtype")

        if copy:
            data = column.as_column(data).copy()
        kwargs = _setdefault_name(data, name=name)
        if isinstance(data, np.ndarray) and data.dtype.kind == "M":
            data = column.as_column(data)
        elif isinstance(data, pd.DatetimeIndex):
            data = column.as_column(data.values)
        elif isinstance(data, (list, tuple)):
            data = column.as_column(np.array(data, dtype=dtype))
        super().__init__(data, **kwargs)

    @property
    def year(self):
        """
        The year of the datetime.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="Y"))
        >>> datetime_index
        DatetimeIndex(['2000-12-31', '2001-12-31', '2002-12-31'], dtype='datetime64[ns]')
        >>> datetime_index.year
        Int16Index([2000, 2001, 2002], dtype='int16')
        """  # noqa: E501
        return self._get_dt_field("year")

    @property
    def month(self):
        """
        The month as January=1, December=12.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="M"))
        >>> datetime_index
        DatetimeIndex(['2000-01-31', '2000-02-29', '2000-03-31'], dtype='datetime64[ns]')
        >>> datetime_index.month
        Int16Index([1, 2, 3], dtype='int16')
        """  # noqa: E501
        return self._get_dt_field("month")

    @property
    def day(self):
        """
        The day of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="D"))
        >>> datetime_index
        DatetimeIndex(['2000-01-01', '2000-01-02', '2000-01-03'], dtype='datetime64[ns]')
        >>> datetime_index.day
        Int16Index([1, 2, 3], dtype='int16')
        """  # noqa: E501
        return self._get_dt_field("day")

    @property
    def hour(self):
        """
        The hours of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="h"))
        >>> datetime_index
        DatetimeIndex(['2000-01-01 00:00:00', '2000-01-01 01:00:00',
                    '2000-01-01 02:00:00'],
                    dtype='datetime64[ns]')
        >>> datetime_index.hour
        Int16Index([0, 1, 2], dtype='int16')
        """
        return self._get_dt_field("hour")

    @property
    def minute(self):
        """
        The minutes of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="T"))
        >>> datetime_index
        DatetimeIndex(['2000-01-01 00:00:00', '2000-01-01 00:01:00',
                    '2000-01-01 00:02:00'],
                    dtype='datetime64[ns]')
        >>> datetime_index.minute
        Int16Index([0, 1, 2], dtype='int16')
        """
        return self._get_dt_field("minute")

    @property
    def second(self):
        """
        The seconds of the datetime.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2000-01-01",
        ...             periods=3, freq="s"))
        >>> datetime_index
        DatetimeIndex(['2000-01-01 00:00:00', '2000-01-01 00:00:01',
                    '2000-01-01 00:00:02'],
                    dtype='datetime64[ns]')
        >>> datetime_index.second
        Int16Index([0, 1, 2], dtype='int16')
        """
        return self._get_dt_field("second")

    @property
    def weekday(self):
        """
        The day of the week with Monday=0, Sunday=6.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2016-12-31",
        ...     "2017-01-08", freq="D"))
        >>> datetime_index
        DatetimeIndex(['2016-12-31', '2017-01-01', '2017-01-02', '2017-01-03',
                    '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07',
                    '2017-01-08'],
                    dtype='datetime64[ns]')
        >>> datetime_index.weekday
        Int16Index([5, 6, 0, 1, 2, 3, 4, 5, 6], dtype='int16')
        """
        return self._get_dt_field("weekday")

    @property
    def dayofweek(self):
        """
        The day of the week with Monday=0, Sunday=6.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2016-12-31",
        ...     "2017-01-08", freq="D"))
        >>> datetime_index
        DatetimeIndex(['2016-12-31', '2017-01-01', '2017-01-02', '2017-01-03',
                    '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07',
                    '2017-01-08'],
                    dtype='datetime64[ns]')
        >>> datetime_index.dayofweek
        Int16Index([5, 6, 0, 1, 2, 3, 4, 5, 6], dtype='int16')
        """
        return self._get_dt_field("weekday")

    @property
    def dayofyear(self):
        """
        The day of the year, from 1-365 in non-leap years and
        from 1-366 in leap years.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2016-12-31",
        ...     "2017-01-08", freq="D"))
        >>> datetime_index
        DatetimeIndex(['2016-12-31', '2017-01-01', '2017-01-02', '2017-01-03',
                    '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07',
                    '2017-01-08'],
                    dtype='datetime64[ns]')
        >>> datetime_index.dayofyear
        Int16Index([366, 1, 2, 3, 4, 5, 6, 7, 8], dtype='int16')
        """
        return self._get_dt_field("day_of_year")

    @property
    def day_of_year(self):
        """
        The day of the year, from 1-365 in non-leap years and
        from 1-366 in leap years.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> datetime_index = cudf.Index(pd.date_range("2016-12-31",
        ...     "2017-01-08", freq="D"))
        >>> datetime_index
        DatetimeIndex(['2016-12-31', '2017-01-01', '2017-01-02', '2017-01-03',
                    '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07',
                    '2017-01-08'],
                    dtype='datetime64[ns]')
        >>> datetime_index.day_of_year
        Int16Index([366, 1, 2, 3, 4, 5, 6, 7, 8], dtype='int16')
        """
        return self._get_dt_field("day_of_year")

    def to_pandas(self):
        nanos = self._values.astype("datetime64[ns]")
        return pd.DatetimeIndex(nanos.to_pandas(), name=self.name)

    def _get_dt_field(self, field):
        out_column = self._values.get_dt_field(field)
        # column.column_empty_like always returns a Column object
        # but we need a NumericalColumn for GenericIndex..
        # how should this be handled?
        out_column = column.build_column(
            data=out_column.base_data,
            dtype=out_column.dtype,
            mask=out_column.base_mask,
            offset=out_column.offset,
        )
        return as_index(out_column, name=self.name)


class TimedeltaIndex(GenericIndex):
    """
    Immutable, ordered and sliceable sequence of timedelta64 data,
    represented internally as int64.

    Parameters
    ----------
    data : array-like (1-dimensional), optional
        Optional datetime-like data to construct index with.
    unit : str, optional
        This is not yet supported
    copy : bool
        Make a copy of input.
    freq : str, optional
        This is not yet supported
    closed : str, optional
        This is not yet supported
    dtype : str or numpy.dtype, optional
        Data type for the output Index. If not specified, the
        default dtype will be ``timedelta64[ns]``.
    name : object
        Name to be stored in the index.

    Returns
    -------
    TimedeltaIndex

    Examples
    --------
    >>> import cudf
    >>> cudf.TimedeltaIndex([1132223, 2023232, 342234324, 4234324],
    ...     dtype='timedelta64[ns]')
    TimedeltaIndex(['00:00:00.001132', '00:00:00.002023', '00:00:00.342234',
                    '00:00:00.004234'],
                dtype='timedelta64[ns]')
    >>> cudf.TimedeltaIndex([1, 2, 3, 4], dtype='timedelta64[s]',
    ...     name="delta-index")
    TimedeltaIndex(['00:00:01', '00:00:02', '00:00:03', '00:00:04'],
                dtype='timedelta64[s]', name='delta-index')
    """

    def __init__(
        self,
        data=None,
        unit=None,
        freq=None,
        closed=None,
        dtype="timedelta64[ns]",
        copy=False,
        name=None,
    ):

        if freq is not None:
            raise NotImplementedError("freq is not yet supported")

        if unit is not None:
            raise NotImplementedError(
                "unit is not yet supported, alternatively "
                "dtype parameter is supported"
            )

        if copy:
            data = column.as_column(data).copy()
        kwargs = _setdefault_name(data, name=name)
        if isinstance(data, np.ndarray) and data.dtype.kind == "m":
            data = column.as_column(data)
        elif isinstance(data, pd.TimedeltaIndex):
            data = column.as_column(data.values)
        elif isinstance(data, (list, tuple)):
            data = column.as_column(np.array(data, dtype=dtype))
        super().__init__(data, **kwargs)

    def to_pandas(self):
        return pd.TimedeltaIndex(
            self._values.to_pandas(),
            name=self.name,
            unit=self._values.time_unit,
        )

    @property
    def days(self):
        """
        Number of days for each element.
        """
        return as_index(arbitrary=self._values.days, name=self.name)

    @property
    def seconds(self):
        """
        Number of seconds (>= 0 and less than 1 day) for each element.
        """
        return as_index(arbitrary=self._values.seconds, name=self.name)

    @property
    def microseconds(self):
        """
        Number of microseconds (>= 0 and less than 1 second) for each element.
        """
        return as_index(arbitrary=self._values.microseconds, name=self.name)

    @property
    def nanoseconds(self):
        """
        Number of nanoseconds (>= 0 and less than 1 microsecond) for each
        element.
        """
        return as_index(arbitrary=self._values.nanoseconds, name=self.name)

    @property
    def components(self):
        """
        Return a dataframe of the components (days, hours, minutes,
        seconds, milliseconds, microseconds, nanoseconds) of the Timedeltas.
        """
        return self._values.components()

    @property
    def inferred_freq(self):
        raise NotImplementedError("inferred_freq is not yet supported")


class CategoricalIndex(GenericIndex):
    """
    A categorical of orderable values that represent the indices of another
    Column

    Parameters
    ----------
    data : array-like (1-dimensional)
        The values of the categorical. If categories are given,
        values not in categories will be replaced with None/NaN.
    categories : list-like, optional
        The categories for the categorical. Items need to be unique.
        If the categories are not given here (and also not in dtype),
        they will be inferred from the data.
    ordered : bool, optional
        Whether or not this categorical is treated as an ordered categorical.
        If not given here or in dtype, the resulting categorical will be
        unordered.
    dtype : CategoricalDtype or “category”, optional
        If CategoricalDtype, cannot be used together with categories or
        ordered.
    copy : bool, default False
        Make a copy of input.
    name : object, optional
        Name to be stored in the index.

    Returns
    -------
    CategoricalIndex

    Examples
    --------
    >>> import cudf
    >>> import pandas as pd
    >>> cudf.CategoricalIndex(
    ... data=[1, 2, 3, 4], categories=[1, 2], ordered=False, name="a")
    CategoricalIndex([1, 2, <NA>, <NA>], categories=[1, 2], ordered=False, name='a', dtype='category', name='a')

    >>> cudf.CategoricalIndex(
    ... data=[1, 2, 3, 4], dtype=pd.CategoricalDtype([1, 2, 3]), name="a")
    CategoricalIndex([1, 2, 3, <NA>], categories=[1, 2, 3], ordered=False, name='a', dtype='category', name='a')
    """  # noqa: E501

    def __init__(
        self,
        data=None,
        categories=None,
        ordered=None,
        dtype=None,
        copy=False,
        name=None,
    ):
        if isinstance(dtype, (pd.CategoricalDtype, cudf.CategoricalDtype)):
            if categories is not None or ordered is not None:
                raise ValueError(
                    "Cannot specify `categories` or "
                    "`ordered` together with `dtype`."
                )
        if copy:
            data = column.as_column(data, dtype=dtype).copy(deep=True)
        kwargs = _setdefault_name(data, name=name)
        if isinstance(data, CategoricalColumn):
            data = data
        elif isinstance(data, pd.Series) and (
            is_categorical_dtype(data.dtype)
        ):
            codes_data = column.as_column(data.cat.codes.values)
            data = column.build_categorical_column(
                categories=data.cat.categories,
                codes=codes_data,
                ordered=data.cat.ordered,
            )
        elif isinstance(data, (pd.Categorical, pd.CategoricalIndex)):
            codes_data = column.as_column(data.codes)
            data = column.build_categorical_column(
                categories=data.categories,
                codes=codes_data,
                ordered=data.ordered,
            )
        else:
            data = column.as_column(
                data, dtype="category" if dtype is None else dtype
            )
            # dtype has already been taken care
            dtype = None

        if categories is not None:
            data.cat().set_categories(
                categories, ordered=ordered, inplace=True
            )
        elif isinstance(dtype, (pd.CategoricalDtype, cudf.CategoricalDtype)):
            data.cat().set_categories(
                dtype.categories, ordered=ordered, inplace=True
            )
        elif ordered is True and data.ordered is False:
            data.cat().as_ordered(inplace=True)
        elif ordered is False and data.ordered is True:
            data.cat().as_unordered(inplace=True)

        super().__init__(data, **kwargs)

    @property
    def codes(self):
        """
        The category codes of this categorical.
        """
        return self._values.cat().codes

    @property
    def categories(self):
        """
        The categories of this categorical.
        """
        return self._values.cat().categories


def interval_range(
    start=None, end=None, periods=None, freq=None, name=None, closed="right",
) -> "IntervalIndex":
    """
    Returns a fixed frequency IntervalIndex.

    Parameters
    ----------
    start : numeric, default None
        Left bound for generating intervals.
    end : numeric , default None
        Right bound for generating intervals.
    periods : int, default None
        Number of periods to generate
    freq : numeric, default None
        The length of each interval. Must be consistent
        with the type of start and end
    name : str, default None
        Name of the resulting IntervalIndex.
    closed : {"left", "right", "both", "neither"}, default "right"
        Whether the intervals are closed on the left-side, right-side,
        both or neither.

    Returns
    -------
    IntervalIndex

    Examples
    --------
    >>> import cudf
    >>> import pandas as pd
    >>> cudf.interval_range(start=0,end=5)
    IntervalIndex([(0, 0], (1, 1], (2, 2], (3, 3], (4, 4], (5, 5]],
    ...closed='right',dtype='interval')
    >>> cudf.interval_range(start=0,end=10, freq=2,closed='left')
    IntervalIndex([[0, 2), [2, 4), [4, 6), [6, 8), [8, 10)],
    ...closed='left',dtype='interval')
    >>> cudf.interval_range(start=0,end=10, periods=3,closed='left')
    ...IntervalIndex([[0.0, 3.3333333333333335),
            [3.3333333333333335, 6.666666666666667),
            [6.666666666666667, 10.0)],
            closed='left',
            dtype='interval')
    """
    if freq and periods and start and end:
        raise ValueError(
            "Of the four parameters: start, end, periods, and "
            "freq, exactly three must be specified"
        )
    args = [
        cudf.Scalar(x) if x is not None else None
        for x in (start, end, freq, periods)
    ]
    if any(
        not _is_non_decimal_numeric_dtype(x.dtype) if x is not None else False
        for x in args
    ):
        raise ValueError("start, end, periods, freq must be numeric values.")
    *rargs, periods = args
    common_dtype = find_common_type([x.dtype for x in rargs if x])
    start, end, freq = rargs
    periods = periods.astype("int64") if periods is not None else None

    if periods and not freq:
        # if statement for mypy to pass
        if end is not None and start is not None:
            # divmod only supported on host side scalars
            quotient, remainder = divmod((end - start).value, periods.value)
            if remainder:
                freq_step = cudf.Scalar((end - start) / periods)
            else:
                freq_step = cudf.Scalar(quotient)
            if start.dtype != freq_step.dtype:
                start = start.astype(freq_step.dtype)
            bin_edges = sequence(
                size=periods + 1,
                init=start.device_value,
                step=freq_step.device_value,
            )
            left_col = bin_edges[:-1]
            right_col = bin_edges[1:]
    elif freq and periods:
        if end:
            start = end - (freq * periods)
        if start:
            end = freq * periods + start
        if end is not None and start is not None:
            left_col = arange(
                start.value, end.value, freq.value, dtype=common_dtype
            )
            end = end + 1
            start = start + freq
            right_col = arange(
                start.value, end.value, freq.value, dtype=common_dtype
            )
    elif freq and not periods:
        if end is not None and start is not None:
            end = end - freq + 1
            left_col = arange(
                start.value, end.value, freq.value, dtype=common_dtype
            )
            end = end + freq + 1
            start = start + freq
            right_col = arange(
                start.value, end.value, freq.value, dtype=common_dtype
            )
    elif start is not None and end is not None:
        # if statements for mypy to pass
        if freq:
            left_col = arange(
                start.value, end.value, freq.value, dtype=common_dtype
            )
        else:
            left_col = arange(start.value, end.value, dtype=common_dtype)
        start = start + 1
        end = end + 1
        if freq:
            right_col = arange(
                start.value, end.value, freq.value, dtype=common_dtype
            )
        else:
            right_col = arange(start.value, end.value, dtype=common_dtype)
    else:
        raise ValueError(
            "Of the four parameters: start, end, periods, and "
            "freq, at least two must be specified"
        )
    if len(right_col) == 0 or len(left_col) == 0:
        dtype = IntervalDtype("int64", closed)
        data = column.column_empty_like_same_mask(left_col, dtype)
        return cudf.IntervalIndex(data, closed=closed)

    interval_col = column.build_interval_column(
        left_col, right_col, closed=closed
    )
    return IntervalIndex(interval_col)


class IntervalIndex(GenericIndex):
    """
    Immutable index of intervals that are closed on the same side.

    Parameters
    ----------
    data : array-like (1-dimensional)
        Array-like containing Interval objects from which to build the
        IntervalIndex.
    closed : {"left", "right", "both", "neither"}, default "right"
        Whether the intervals are closed on the left-side, right-side,
        both or neither.
    dtype : dtype or None, default None
        If None, dtype will be inferred.
    copy : bool, default False
        Copy the input data.
    name : object, optional
        Name to be stored in the index.

    Returns
    -------
    IntervalIndex
    """

    def __init__(
        self, data, closed=None, dtype=None, copy=False, name=None,
    ):
        if copy:
            data = column.as_column(data, dtype=dtype).copy()
        kwargs = _setdefault_name(data, name=name)
        if isinstance(data, IntervalColumn):
            data = data
        elif isinstance(data, pd.Series) and (is_interval_dtype(data.dtype)):
            data = column.as_column(data, data.dtype)
        elif isinstance(data, (pd._libs.interval.Interval, pd.IntervalIndex)):
            data = column.as_column(data, dtype=dtype,)
        elif not data:
            dtype = IntervalDtype("int64", closed)
            data = column.column_empty_like_same_mask(
                column.as_column(data), dtype
            )
        else:
            data = column.as_column(data)
            data.dtype.closed = closed

        super().__init__(data, **kwargs)

    def from_breaks(breaks, closed="right", name=None, copy=False, dtype=None):
        """
        Construct an IntervalIndex from an array of splits.

        Parameters
        ---------
        breaks : array-like (1-dimensional)
            Left and right bounds for each interval.
        closed : {"left", "right", "both", "neither"}, default "right"
            Whether the intervals are closed on the left-side, right-side,
            both or neither.
        copy : bool, default False
            Copy the input data.
        name : object, optional
            Name to be stored in the index.
        dtype : dtype or None, default None
            If None, dtype will be inferred.

        Returns
        -------
        IntervalIndex

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> cudf.IntervalIndex.from_breaks([0, 1, 2, 3])
        IntervalIndex([(0, 1], (1, 2], (2, 3]],
                    closed='right',
                    dtype='interval[int64]')
        """
        if copy:
            breaks = column.as_column(breaks, dtype=dtype).copy()
        left_col = breaks[:-1:]
        right_col = breaks[+1::]

        interval_col = column.build_interval_column(
            left_col, right_col, closed=closed
        )

        return IntervalIndex(interval_col, name=name)


class StringIndex(GenericIndex):
    """String defined indices into another Column

    Attributes
    ----------
    _values: A StringColumn object or NDArray of strings
    name: A string
    """

    def __init__(self, values, copy=False, **kwargs):
        kwargs = _setdefault_name(values, **kwargs)
        if isinstance(values, StringColumn):
            values = values.copy(deep=copy)
        elif isinstance(values, StringIndex):
            values = values._values.copy(deep=copy)
        else:
            values = column.as_column(values, dtype="str")
            if not is_string_dtype(values.dtype):
                raise ValueError(
                    "Couldn't create StringIndex from passed in object"
                )

        super().__init__(values, **kwargs)

    def to_pandas(self):
        return pd.Index(self.to_array(), name=self.name, dtype="object")

    def take(self, indices):
        return self._values[indices]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self._values.to_array()},"
            f" dtype='object'"
            + (
                f", name={pd.io.formats.printing.default_pprint(self.name)}"
                if self.name is not None
                else ""
            )
            + ")"
        )

    @copy_docstring(StringMethods.__init__)  # type: ignore
    @property
    def str(self):
        return StringMethods(column=self._values, parent=self)

    def _clean_nulls_from_index(self):
        """
        Convert all na values(if any) in Index object
        to `<NA>` as a preprocessing step to `__repr__` methods.
        """
        if self._values.has_nulls:
            return self.fillna(cudf._NA_REP)
        else:
            return self


def as_index(arbitrary, **kwargs) -> BaseIndex:
    """Create an Index from an arbitrary object

    Currently supported inputs are:

    * ``Column``
    * ``Buffer``
    * ``Series``
    * ``Index``
    * numba device array
    * numpy array
    * pyarrow array
    * pandas.Categorical

    Returns
    -------
    result : subclass of Index
        - CategoricalIndex for Categorical input.
        - DatetimeIndex for Datetime input.
        - GenericIndex for all other inputs.
    """
    kwargs = _setdefault_name(arbitrary, **kwargs)
    if isinstance(arbitrary, cudf.MultiIndex):
        return arbitrary
    elif isinstance(arbitrary, BaseIndex):
        if arbitrary.name == kwargs["name"]:
            return arbitrary
        idx = arbitrary.copy(deep=False)
        idx.rename(kwargs["name"], inplace=True)
        return idx
    elif isinstance(arbitrary, NumericalColumn):
        try:
            return _dtype_to_index[arbitrary.dtype.type](arbitrary, **kwargs)
        except KeyError:
            return GenericIndex(arbitrary, **kwargs)
    elif isinstance(arbitrary, StringColumn):
        return StringIndex(arbitrary, **kwargs)
    elif isinstance(arbitrary, DatetimeColumn):
        return DatetimeIndex(arbitrary, **kwargs)
    elif isinstance(arbitrary, TimeDeltaColumn):
        return TimedeltaIndex(arbitrary, **kwargs)
    elif isinstance(arbitrary, CategoricalColumn):
        return CategoricalIndex(arbitrary, **kwargs)
    elif isinstance(arbitrary, IntervalColumn):
        return IntervalIndex(arbitrary, **kwargs)
    elif isinstance(arbitrary, cudf.Series):
        return as_index(arbitrary._column, **kwargs)
    elif isinstance(arbitrary, pd.RangeIndex):
        return RangeIndex(start=arbitrary.start, stop=arbitrary.stop, **kwargs)
    elif isinstance(arbitrary, pd.MultiIndex):
        return cudf.MultiIndex.from_pandas(arbitrary)
    elif isinstance(arbitrary, cudf.DataFrame):
        return cudf.MultiIndex(source_data=arbitrary)
    elif isinstance(arbitrary, range):
        return RangeIndex(arbitrary, **kwargs)
    return as_index(
        column.as_column(arbitrary, dtype=kwargs.get("dtype", None)), **kwargs
    )


_dtype_to_index: Dict[Any, Type[BaseIndex]] = {
    np.int8: Int8Index,
    np.int16: Int16Index,
    np.int32: Int32Index,
    np.int64: Int64Index,
    np.uint8: UInt8Index,
    np.uint16: UInt16Index,
    np.uint32: UInt32Index,
    np.uint64: UInt64Index,
    np.float32: Float32Index,
    np.float64: Float64Index,
}


def _setdefault_name(values, **kwargs):
    if kwargs.get("name") is None:
        kwargs["name"] = getattr(values, "name", None)
    return kwargs


class IndexMeta(type):
    """Custom metaclass for Index that overrides instance/subclass tests."""

    def __instancecheck__(self, instance):
        return isinstance(instance, BaseIndex)

    def __subclasscheck__(self, subclass):
        return issubclass(subclass, BaseIndex)


class Index(BaseIndex, metaclass=IndexMeta):
    """The basic object storing row labels for all cuDF objects.

    Parameters
    ----------
    data : array-like (1-dimensional)/ DataFrame
        If it is a DataFrame, it will return a MultiIndex
    dtype : NumPy dtype (default: object)
        If dtype is None, we find the dtype that best fits the data.
    copy : bool
        Make a copy of input data.
    name : object
        Name to be stored in the index.
    tupleize_cols : bool (default: True)
        When True, attempt to create a MultiIndex if possible.
        tupleize_cols == False is not yet supported.

    Returns
    -------
    Index
        cudf Index

    Warnings
    --------
    This class should not be subclassed. It is designed as a factory for
    different subclasses of :class:`BaseIndex` depending on the provided input.
    If you absolutely must, and if you're intimately familiar with the
    internals of cuDF, subclass :class:`BaseIndex` instead.

    Examples
    --------
    >>> import cudf
    >>> cudf.Index([1, 2, 3], dtype="uint64", name="a")
    UInt64Index([1, 2, 3], dtype='uint64', name='a')

    >>> cudf.Index(cudf.DataFrame({"a":[1, 2], "b":[2, 3]}))
    MultiIndex([(1, 2),
                (2, 3)],
                names=['a', 'b'])
    """

    def __new__(
        cls,
        data=None,
        dtype=None,
        copy=False,
        name=None,
        tupleize_cols=True,
        **kwargs,
    ):
        assert cls is Index, (
            "Index cannot be subclassed, extend BaseIndex " "instead."
        )
        if tupleize_cols is not True:
            raise NotImplementedError(
                "tupleize_cols != True is not yet supported"
            )

        return as_index(data, copy=copy, dtype=dtype, name=name, **kwargs)
