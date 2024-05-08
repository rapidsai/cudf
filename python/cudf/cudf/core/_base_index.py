# Copyright (c) 2021-2024, NVIDIA CORPORATION.

from __future__ import annotations

import pickle
import warnings
from collections.abc import Generator
from functools import cached_property
from typing import Any, Literal, Set, Tuple

import pandas as pd
from typing_extensions import Self

import cudf
from cudf._lib.copying import _gather_map_is_valid, gather
from cudf._lib.stream_compaction import (
    apply_boolean_mask,
    drop_duplicates,
    drop_nulls,
)
from cudf._lib.types import size_type_dtype
from cudf.api.extensions import no_default
from cudf.api.types import (
    is_bool_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_scalar,
    is_signed_integer_dtype,
    is_unsigned_integer_dtype,
)
from cudf.core.abc import Serializable
from cudf.core.column import ColumnBase, column
from cudf.core.column_accessor import ColumnAccessor
from cudf.errors import MixedTypeError
from cudf.utils import ioutils
from cudf.utils.dtypes import can_convert_to_column, is_mixed_with_object_dtype
from cudf.utils.utils import _is_same_name


class BaseIndex(Serializable):
    """Base class for all cudf Index types."""

    _accessors: Set[Any] = set()
    _data: ColumnAccessor

    @property
    def _columns(self) -> Tuple[Any, ...]:
        raise NotImplementedError

    @cached_property
    def _values(self) -> ColumnBase:
        raise NotImplementedError

    def copy(self, deep: bool = True) -> Self:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def size(self):
        # The size of an index is always its length irrespective of dimension.
        return len(self)

    def astype(self, dtype, copy: bool = True):
        """Create an Index with values cast to dtypes.

        The class of a new Index is determined by dtype. When conversion is
        impossible, a ValueError exception is raised.

        Parameters
        ----------
        dtype : :class:`numpy.dtype`
            Use a :class:`numpy.dtype` to cast entire Index object to.
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
        Index([1, 2, 3], dtype='int64')
        >>> index.astype('float64')
        Index([1.0, 2.0, 3.0], dtype='float64')
        """
        raise NotImplementedError

    def argsort(self, *args, **kwargs):
        """Return the integer indices that would sort the index.

        Parameters vary by subclass.
        """
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def empty(self):
        return self.size == 0

    @property
    def is_unique(self):
        """Return if the index has unique values."""
        raise NotImplementedError

    def memory_usage(self, deep=False):
        """Return the memory usage of an object.

        Parameters
        ----------
        deep : bool
            The deep parameter is ignored and is only included for pandas
            compatibility.

        Returns
        -------
        The total bytes used.
        """
        raise NotImplementedError

    def tolist(self):  # noqa: D102
        raise TypeError(
            "cuDF does not support conversion to host memory "
            "via the `tolist()` method. Consider using "
            "`.to_arrow().to_pylist()` to construct a Python list."
        )

    to_list = tolist

    @property
    def name(self):
        """Returns the name of the Index."""
        raise NotImplementedError

    @property  # type: ignore
    def ndim(self):  # noqa: D401
        """Number of dimensions of the underlying data, by definition 1."""
        return 1

    def equals(self, other):
        """
        Determine if two Index objects contain the same elements.

        Returns
        -------
        out: bool
            True if "other" is an Index and it has the same elements
            as calling index; False otherwise.
        """
        raise NotImplementedError

    def shift(self, periods=1, freq=None):
        """Not yet implemented"""
        raise NotImplementedError

    @property
    def shape(self):
        """Get a tuple representing the dimensionality of the data."""
        return (len(self),)

    @property
    def str(self):
        """Not yet implemented."""
        raise NotImplementedError

    @property
    def values(self):
        raise NotImplementedError

    def get_indexer(self, target, method=None, limit=None, tolerance=None):
        """
        Compute indexer and mask for new index given the current index.

        The indexer should be then used as an input to ndarray.take to align
        the current data to the new index.

        Parameters
        ----------
        target : Index
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
            ``abs(index[loc] - target) <= tolerance``.

        Returns
        -------
        cupy.ndarray
            Integers from 0 to n - 1 indicating that the index at these
            positions matches the corresponding target values.
            Missing values in the target are marked by -1.

        Examples
        --------
        >>> import cudf
        >>> index = cudf.Index(['c', 'a', 'b'])
        >>> index
        Index(['c', 'a', 'b'], dtype='object')
        >>> index.get_indexer(['a', 'b', 'x'])
        array([ 1,  2, -1], dtype=int32)
        """
        raise NotImplementedError

    def get_loc(self, key):
        """
        Get integer location, slice or boolean mask for requested label.

        Parameters
        ----------
        key : label

        Returns
        -------
        int or slice or boolean mask
            - If result is unique, return integer index
            - If index is monotonic, loc is returned as a slice object
            - Otherwise, a boolean mask is returned

        Examples
        --------
        >>> import cudf
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

        **MultiIndex**

        >>> multi_index = cudf.MultiIndex.from_tuples([('a', 'd'), ('b', 'e'), ('b', 'f')])
        >>> multi_index
        MultiIndex([('a', 'd'),
                    ('b', 'e'),
                    ('b', 'f')],
                )
        >>> multi_index.get_loc('b')
        slice(1, 3, None)
        >>> multi_index.get_loc(('b', 'e'))
        1
        """  # noqa: E501

    def max(self):
        """The maximum value of the index."""
        raise NotImplementedError

    def min(self):
        """The minimum value of the index."""
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError()

    def __contains__(self, item):
        hash(item)
        return item in self._values

    def _copy_type_metadata(
        self, other: Self, *, override_dtypes=None
    ) -> Self:
        raise NotImplementedError

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
        cudf.MultiIndex.get_level_values : Get values for
            a level of a MultiIndex.

        Notes
        -----
        For Index, level should be 0, since there are no multiple levels.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index(["a", "b", "c"])
        >>> idx.get_level_values(0)
        Index(['a', 'b', 'c'], dtype='object')
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
        # Dispatch deserialization to the appropriate index type in case
        # deserialization is ever attempted with the base class directly.
        idx_type = pickle.loads(header["type-serialized"])
        return idx_type.deserialize(header, frames)

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

    def _clean_nulls_from_index(self):
        """
        Convert all na values(if any) in Index object
        to `<NA>` as a preprocessing step to `__repr__` methods.

        This will involve changing type of Index object
        to string dtype but it is the responsibility of the `__repr__`
        methods using this method to replace or handle representation
        of the actual types correctly.
        """
        raise NotImplementedError

    @property
    def is_monotonic_increasing(self):
        """Return boolean if values in the object are monotonically increasing.

        Returns
        -------
        bool
        """
        raise NotImplementedError

    @property
    def is_monotonic_decreasing(self):
        """Return boolean if values in the object are monotonically decreasing.

        Returns
        -------
        bool
        """
        raise NotImplementedError

    @property
    def hasnans(self):
        """
        Return True if there are any NaNs or nulls.

        Returns
        -------
        out : bool
            If Series has at least one NaN or null value, return True,
            if not return False.

        Examples
        --------
        >>> import cudf
        >>> import numpy as np
        >>> index = cudf.Index([1, 2, np.nan, 3, 4], nan_as_null=False)
        >>> index
        Index([1.0, 2.0, nan, 3.0, 4.0], dtype='float64')
        >>> index.hasnans
        True

        `hasnans` returns `True` for the presence of any `NA` values:

        >>> index = cudf.Index([1, 2, None, 3, 4])
        >>> index
        Index([1, 2, <NA>, 3, 4], dtype='int64')
        >>> index.hasnans
        True
        """
        raise NotImplementedError

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
        cudf.Index.rename : Able to set new names without level.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index([1, 2, 3, 4])
        >>> idx
        Index([1, 2, 3, 4], dtype='int64')
        >>> idx.set_names('quarter')
        Index([1, 2, 3, 4], dtype='int64', name='quarter')
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

    @property
    def has_duplicates(self):
        return not self.is_unique

    def where(self, cond, other=None, inplace=False):
        """
        Replace values where the condition is False.

        The replacement is taken from other.

        Parameters
        ----------
        cond : bool array-like with the same length as self
            Condition to select the values on.
        other : scalar, or array-like, default None
            Replacement if the condition is False.

        Returns
        -------
        cudf.Index
            A copy of self with values replaced from other
            where the condition is False.
        """
        raise NotImplementedError

    def factorize(self, sort: bool = False, use_na_sentinel: bool = True):
        raise NotImplementedError

    def union(self, other, sort=None):
        """
        Form the union of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        sort : bool or None, default None
            Whether to sort the resulting Index.

            * None : Sort the result, except when

              1. `self` and `other` are equal.
              2. `self` or `other` has length 0.

            * False : do not sort the result.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        union : Index

        Examples
        --------
        Union of an Index
        >>> import cudf
        >>> import pandas as pd
        >>> idx1 = cudf.Index([1, 2, 3, 4])
        >>> idx2 = cudf.Index([3, 4, 5, 6])
        >>> idx1.union(idx2)
        Index([1, 2, 3, 4, 5, 6], dtype='int64')

        MultiIndex case

        >>> idx1 = cudf.MultiIndex.from_pandas(
        ...    pd.MultiIndex.from_arrays(
        ...         [[1, 1, 2, 2], ["Red", "Blue", "Red", "Blue"]]
        ...    )
        ... )
        >>> idx1
        MultiIndex([(1,  'Red'),
                    (1, 'Blue'),
                    (2,  'Red'),
                    (2, 'Blue')],
                   )
        >>> idx2 = cudf.MultiIndex.from_pandas(
        ...    pd.MultiIndex.from_arrays(
        ...         [[3, 3, 2, 2], ["Red", "Green", "Red", "Green"]]
        ...    )
        ... )
        >>> idx2
        MultiIndex([(3,   'Red'),
                    (3, 'Green'),
                    (2,   'Red'),
                    (2, 'Green')],
                   )
        >>> idx1.union(idx2)
        MultiIndex([(1,  'Blue'),
                    (1,   'Red'),
                    (2,  'Blue'),
                    (2, 'Green'),
                    (2,   'Red'),
                    (3, 'Green'),
                    (3,   'Red')],
                   )
        >>> idx1.union(idx2, sort=False)
        MultiIndex([(1,   'Red'),
                    (1,  'Blue'),
                    (2,   'Red'),
                    (2,  'Blue'),
                    (3,   'Red'),
                    (3, 'Green'),
                    (2, 'Green')],
                   )
        """
        if not isinstance(other, BaseIndex):
            other = cudf.Index(other, name=self.name)

        if sort not in {None, False, True}:
            raise ValueError(
                f"The 'sort' keyword only takes the values of "
                f"[None, False, True]; {sort} was passed."
            )

        if cudf.get_option("mode.pandas_compatible"):
            if (
                is_bool_dtype(self.dtype) and not is_bool_dtype(other.dtype)
            ) or (
                not is_bool_dtype(self.dtype) and is_bool_dtype(other.dtype)
            ):
                # Bools + other types will result in mixed type.
                # This is not yet consistent in pandas and specific to APIs.
                raise MixedTypeError("Cannot perform union with mixed types")
            if (
                is_signed_integer_dtype(self.dtype)
                and is_unsigned_integer_dtype(other.dtype)
            ) or (
                is_unsigned_integer_dtype(self.dtype)
                and is_signed_integer_dtype(other.dtype)
            ):
                # signed + unsigned types will result in
                # mixed type for union in pandas.
                raise MixedTypeError("Cannot perform union with mixed types")

        if not len(other) or self.equals(other):
            common_dtype = cudf.utils.dtypes.find_common_type(
                [self.dtype, other.dtype]
            )
            res = self._get_reconciled_name_object(other).astype(common_dtype)
            if sort:
                return res.sort_values()
            return res
        elif not len(self):
            common_dtype = cudf.utils.dtypes.find_common_type(
                [self.dtype, other.dtype]
            )
            res = other._get_reconciled_name_object(self).astype(common_dtype)
            if sort:
                return res.sort_values()
            return res

        result = self._union(other, sort=sort)
        result.name = _get_result_name(self.name, other.name)
        return result

    def intersection(self, other, sort=False):
        """
        Form the intersection of two Index objects.

        This returns a new Index with elements common to the index and `other`.

        Parameters
        ----------
        other : Index or array-like
        sort : False or None, default False
            Whether to sort the resulting index.

            * False : do not sort the result.
            * None : sort the result, except when `self` and `other` are equal
              or when the values cannot be compared.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        intersection : Index

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> idx1 = cudf.Index([1, 2, 3, 4])
        >>> idx2 = cudf.Index([3, 4, 5, 6])
        >>> idx1.intersection(idx2)
        Index([3, 4], dtype='int64')

        MultiIndex case

        >>> idx1 = cudf.MultiIndex.from_pandas(
        ...    pd.MultiIndex.from_arrays(
        ...         [[1, 1, 3, 4], ["Red", "Blue", "Red", "Blue"]]
        ...    )
        ... )
        >>> idx2 = cudf.MultiIndex.from_pandas(
        ...    pd.MultiIndex.from_arrays(
        ...         [[1, 1, 2, 2], ["Red", "Blue", "Red", "Blue"]]
        ...    )
        ... )
        >>> idx1
        MultiIndex([(1,  'Red'),
                    (1, 'Blue'),
                    (3,  'Red'),
                    (4, 'Blue')],
                )
        >>> idx2
        MultiIndex([(1,  'Red'),
                    (1, 'Blue'),
                    (2,  'Red'),
                    (2, 'Blue')],
                )
        >>> idx1.intersection(idx2)
        MultiIndex([(1,  'Red'),
                    (1, 'Blue')],
                )
        >>> idx1.intersection(idx2, sort=False)
        MultiIndex([(1,  'Red'),
                    (1, 'Blue')],
                )
        """
        if not can_convert_to_column(other):
            raise TypeError("Input must be Index or array-like")

        if not isinstance(other, BaseIndex):
            other = cudf.Index(
                other,
                name=getattr(other, "name", self.name),
            )

        if sort not in {None, False, True}:
            raise ValueError(
                f"The 'sort' keyword only takes the values of "
                f"[None, False, True]; {sort} was passed."
            )

        if not len(self) or not len(other) or self.equals(other):
            common_dtype = cudf.utils.dtypes._dtype_pandas_compatible(
                cudf.utils.dtypes.find_common_type([self.dtype, other.dtype])
            )

            lhs = self.unique() if self.has_duplicates else self
            rhs = other
            if not len(other):
                lhs, rhs = rhs, lhs

            return lhs._get_reconciled_name_object(rhs).astype(common_dtype)

        res_name = _get_result_name(self.name, other.name)

        if (self._is_boolean() and other._is_numeric()) or (
            self._is_numeric() and other._is_boolean()
        ):
            if isinstance(self, cudf.MultiIndex):
                return self[:0].rename(res_name)
            else:
                return cudf.Index([], name=res_name)

        if self.has_duplicates:
            lhs = self.unique()
        else:
            lhs = self
        if other.has_duplicates:
            rhs = other.unique()
        else:
            rhs = other
        result = lhs._intersection(rhs, sort=sort)
        result.name = res_name
        return result

    def _get_reconciled_name_object(self, other):
        """
        If the result of a set operation will be self,
        return self, unless the name changes, in which
        case make a shallow copy of self.
        """
        name = _get_result_name(self.name, other.name)
        if not _is_same_name(self.name, name):
            return self.rename(name)
        return self

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
        Index([1, 2, <NA>, 4], dtype='int64')
        >>> index.fillna(3)
        Index([1, 2, 3, 4], dtype='int64')
        """
        if downcast is not None:
            raise NotImplementedError(
                "`downcast` parameter is not yet supported"
            )

        return super().fillna(value=value)

    def to_frame(self, index=True, name=no_default):
        """Create a DataFrame with a column containing this Index

        Parameters
        ----------
        index : boolean, default True
            Set the index of the returned DataFrame as the original Index
        name : object, defaults to index.name
            The passed name should substitute for the index name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame containing the original Index data.

        See Also
        --------
        Index.to_series : Convert an Index to a Series.
        Series.to_frame : Convert Series to DataFrame.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index(['Ant', 'Bear', 'Cow'], name='animal')
        >>> idx.to_frame()
               animal
        animal
        Ant       Ant
        Bear     Bear
        Cow       Cow

        By default, the original Index is reused. To enforce a new Index:

        >>> idx.to_frame(index=False)
            animal
        0   Ant
        1  Bear
        2   Cow

        To override the name of the resulting column, specify `name`:

        >>> idx.to_frame(index=False, name='zoo')
            zoo
        0   Ant
        1  Bear
        2   Cow
        """

        if name is no_default:
            col_name = 0 if self.name is None else self.name
        else:
            col_name = name

        return cudf.DataFrame(
            {col_name: self._values}, index=self if index else None
        )

    def to_arrow(self):
        """Convert to a suitable Arrow object."""
        raise NotImplementedError

    def to_cupy(self):
        """Convert to a cupy array."""
        raise NotImplementedError

    def to_numpy(self):
        """Convert to a numpy array."""
        raise NotImplementedError

    def any(self):
        """
        Return whether any elements is True in Index.
        """
        raise NotImplementedError

    def isna(self):
        """
        Detect missing values.

        Return a boolean same-sized object indicating if the values are NA.
        NA values, such as ``None``, `numpy.NAN` or `cudf.NA`, get
        mapped to ``True`` values.
        Everything else get mapped to ``False`` values.

        Returns
        -------
        numpy.ndarray[bool]
            A boolean array to indicate which entries are NA.

        """
        raise NotImplementedError

    def notna(self):
        """
        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to ``True``.
        NA values, such as None or `numpy.NAN`, get mapped to ``False``
        values.

        Returns
        -------
        numpy.ndarray[bool]
            A boolean array to indicate which entries are not NA.
        """
        raise NotImplementedError

    def to_pandas(self, *, nullable: bool = False, arrow_type: bool = False):
        """
        Convert to a Pandas Index.

        Parameters
        ----------
        nullable : bool, Default False
            If ``nullable`` is ``True``, the resulting index will have
            a corresponding nullable Pandas dtype.
            If there is no corresponding nullable Pandas dtype present,
            the resulting dtype will be a regular pandas dtype.
            If ``nullable`` is ``False``, the resulting index will
            either convert null values to ``np.nan`` or ``None``
            depending on the dtype.
        arrow_type : bool, Default False
            Return the Index with a ``pandas.ArrowDtype``

        Notes
        -----
        nullable and arrow_type cannot both be set to ``True``

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index([-3, 10, 15, 20])
        >>> idx
        Index([-3, 10, 15, 20], dtype='int64')
        >>> idx.to_pandas()
        Index([-3, 10, 15, 20], dtype='int64')
        >>> type(idx.to_pandas())
        <class 'pandas.core.indexes.base.Index'>
        >>> type(idx)
        <class 'cudf.core.index.Index'>
        >>> idx.to_pandas(arrow_type=True)
        Index([-3, 10, 15, 20], dtype='int64[pyarrow]')
        """
        raise NotImplementedError

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
        Index([1, 2, 3], dtype='int64')

        Check whether each index value in a list of values.

        >>> idx.isin([1, 4])
        array([ True, False, False])
        """
        # To match pandas behavior, even though only list-like objects are
        # supposed to be passed, only scalars throw errors. Other types (like
        # dicts) just transparently return False (see the implementation of
        # ColumnBase.isin).
        raise NotImplementedError

    def unique(self):
        """
        Return unique values in the index.

        Returns
        -------
        Index without duplicates
        """
        raise NotImplementedError

    def to_series(self, index=None, name=None):
        """
        Create a Series with both index and values equal to the index keys.
        Useful with map for returning an indexer based on an index.

        Parameters
        ----------
        index : Index, optional
            Index of resulting Series. If None, defaults to original index.
        name : str, optional
            Name of resulting Series. If None, defaults to name of original
            index.

        Returns
        -------
        Series
            The dtype will be based on the type of the Index values.
        """
        return cudf.Series._from_data(
            self._data,
            index=self.copy(deep=False) if index is None else index,
            name=self.name if name is None else name,
        )

    @ioutils.doc_to_dlpack()
    def to_dlpack(self):
        """{docstring}"""

        return cudf.io.dlpack.to_dlpack(self)

    def append(self, other):
        """
        Append a collection of Index objects together.

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
        Index([1, 2, 10, 100], dtype='int64')
        >>> other = cudf.Index([200, 400, 50])
        >>> other
        Index([200, 400, 50], dtype='int64')
        >>> idx.append(other)
        Index([1, 2, 10, 100, 200, 400, 50], dtype='int64')

        append accepts list of Index objects

        >>> idx.append([other, other])
        Index([1, 2, 10, 100, 200, 400, 50, 200, 400, 50], dtype='int64')
        """
        raise NotImplementedError

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
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        difference : Index

        Examples
        --------
        >>> import cudf
        >>> idx1 = cudf.Index([2, 1, 3, 4])
        >>> idx1
        Index([2, 1, 3, 4], dtype='int64')
        >>> idx2 = cudf.Index([3, 4, 5, 6])
        >>> idx2
        Index([3, 4, 5, 6], dtype='int64')
        >>> idx1.difference(idx2)
        Index([1, 2], dtype='int64')
        >>> idx1.difference(idx2, sort=False)
        Index([2, 1], dtype='int64')
        """

        if not can_convert_to_column(other):
            raise TypeError("Input must be Index or array-like")

        if sort not in {None, False, True}:
            raise ValueError(
                f"The 'sort' keyword only takes the values "
                f"of [None, False, True]; {sort} was passed."
            )

        other = cudf.Index(other, name=getattr(other, "name", self.name))

        if not len(other):
            res = self._get_reconciled_name_object(other).unique()
            if sort:
                return res.sort_values()
            return res
        elif self.equals(other):
            res = self[:0]._get_reconciled_name_object(other).unique()
            if sort:
                return res.sort_values()
            return res

        res_name = _get_result_name(self.name, other.name)

        if is_mixed_with_object_dtype(self, other) or len(other) == 0:
            difference = self.copy().unique()
            difference.name = res_name
            if sort is True:
                return difference.sort_values()
        else:
            other = other.copy(deep=False)
            difference = cudf.core.index._index_from_data(
                cudf.DataFrame._from_data({"None": self._column.unique()})
                .merge(
                    cudf.DataFrame._from_data({"None": other._column}),
                    how="leftanti",
                    on="None",
                )
                ._data
            )
            difference.name = res_name

            if self.dtype != other.dtype:
                difference = difference.astype(self.dtype)

        if sort in {None, True} and len(other):
            return difference.sort_values()

        return difference

    def is_numeric(self):
        """
        Check if the Index only consists of numeric data.

        .. deprecated:: 23.04
           Use `cudf.api.types.is_any_real_numeric_dtype` instead.

        Returns
        -------
        bool
            Whether or not the Index only consists of numeric data.

        See Also
        --------
        is_boolean : Check if the Index only consists of booleans.
        is_integer : Check if the Index only consists of integers.
        is_floating : Check if the Index is a floating type.
        is_object : Check if the Index is of the object dtype.
        is_categorical : Check if the Index holds categorical data.
        is_interval : Check if the Index holds Interval objects.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index([1.0, 2.0, 3.0, 4.0])
        >>> idx.is_numeric()
        True
        >>> idx = cudf.Index([1, 2, 3, 4.0])
        >>> idx.is_numeric()
        True
        >>> idx = cudf.Index([1, 2, 3, 4])
        >>> idx.is_numeric()
        True
        >>> idx = cudf.Index([1, 2, 3, 4.0, np.nan])
        >>> idx.is_numeric()
        True
        >>> idx = cudf.Index(["Apple", "cold"])
        >>> idx.is_numeric()
        False
        """
        # Do not remove until pandas removes this.
        warnings.warn(
            f"{type(self).__name__}.is_numeric is deprecated. "
            "Use cudf.api.types.is_any_real_numeric_dtype instead",
            FutureWarning,
        )
        return self._is_numeric()

    def _is_numeric(self):
        raise NotImplementedError

    def is_boolean(self):
        """
        Check if the Index only consists of booleans.

        .. deprecated:: 23.04
           Use `cudf.api.types.is_bool_dtype` instead.

        Returns
        -------
        bool
            Whether or not the Index only consists of booleans.

        See Also
        --------
        is_integer : Check if the Index only consists of integers.
        is_floating : Check if the Index is a floating type.
        is_numeric : Check if the Index only consists of numeric data.
        is_object : Check if the Index is of the object dtype.
        is_categorical : Check if the Index holds categorical data.
        is_interval : Check if the Index holds Interval objects.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index([True, False, True])
        >>> idx.is_boolean()
        True
        >>> idx = cudf.Index(["True", "False", "True"])
        >>> idx.is_boolean()
        False
        >>> idx = cudf.Index([1, 2, 3])
        >>> idx.is_boolean()
        False
        """
        # Do not remove until pandas removes this.
        warnings.warn(
            f"{type(self).__name__}.is_boolean is deprecated. "
            "Use cudf.api.types.is_bool_dtype instead",
            FutureWarning,
        )
        return self._is_boolean()

    def _is_boolean(self):
        raise NotImplementedError

    def is_integer(self):
        """
        Check if the Index only consists of integers.

        .. deprecated:: 23.04
           Use `cudf.api.types.is_integer_dtype` instead.

        Returns
        -------
        bool
            Whether or not the Index only consists of integers.

        See Also
        --------
        is_boolean : Check if the Index only consists of booleans.
        is_floating : Check if the Index is a floating type.
        is_numeric : Check if the Index only consists of numeric data.
        is_object : Check if the Index is of the object dtype.
        is_categorical : Check if the Index holds categorical data.
        is_interval : Check if the Index holds Interval objects.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index([1, 2, 3, 4])
        >>> idx.is_integer()
        True
        >>> idx = cudf.Index([1.0, 2.0, 3.0, 4.0])
        >>> idx.is_integer()
        False
        >>> idx = cudf.Index(["Apple", "Mango", "Watermelon"])
        >>> idx.is_integer()
        False
        """
        # Do not remove until pandas removes this.
        warnings.warn(
            f"{type(self).__name__}.is_integer is deprecated. "
            "Use cudf.api.types.is_integer_dtype instead",
            FutureWarning,
        )
        return self._is_integer()

    def _is_integer(self):
        raise NotImplementedError

    def is_floating(self):
        """
        Check if the Index is a floating type.

        The Index may consist of only floats, NaNs, or a mix of floats,
        integers, or NaNs.

        .. deprecated:: 23.04
           Use `cudf.api.types.is_float_dtype` instead.

        Returns
        -------
        bool
            Whether or not the Index only consists of only consists
            of floats, NaNs, or a mix of floats, integers, or NaNs.

        See Also
        --------
        is_boolean : Check if the Index only consists of booleans.
        is_integer : Check if the Index only consists of integers.
        is_numeric : Check if the Index only consists of numeric data.
        is_object : Check if the Index is of the object dtype.
        is_categorical : Check if the Index holds categorical data.
        is_interval : Check if the Index holds Interval objects.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index([1.0, 2.0, 3.0, 4.0])
        >>> idx.is_floating()
        True
        >>> idx = cudf.Index([1.0, 2.0, np.nan, 4.0])
        >>> idx.is_floating()
        True
        >>> idx = cudf.Index([1, 2, 3, 4, np.nan], nan_as_null=False)
        >>> idx.is_floating()
        True
        >>> idx = cudf.Index([1, 2, 3, 4])
        >>> idx.is_floating()
        False
        """
        # Do not remove until pandas removes this.
        warnings.warn(
            f"{type(self).__name__}.is_floating is deprecated. "
            "Use cudf.api.types.is_float_dtype instead",
            FutureWarning,
        )
        return self._is_floating()

    def _is_floating(self):
        raise NotImplementedError

    def is_object(self):
        """
        Check if the Index is of the object dtype.

        .. deprecated:: 23.04
           Use `cudf.api.types.is_object_dtype` instead.

        Returns
        -------
        bool
            Whether or not the Index is of the object dtype.

        See Also
        --------
        is_boolean : Check if the Index only consists of booleans.
        is_integer : Check if the Index only consists of integers.
        is_floating : Check if the Index is a floating type.
        is_numeric : Check if the Index only consists of numeric data.
        is_categorical : Check if the Index holds categorical data.
        is_interval : Check if the Index holds Interval objects.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index(["Apple", "Mango", "Watermelon"])
        >>> idx.is_object()
        True
        >>> idx = cudf.Index(["Watermelon", "Orange", "Apple",
        ...                 "Watermelon"]).astype("category")
        >>> idx.is_object()
        False
        >>> idx = cudf.Index([1.0, 2.0, 3.0, 4.0])
        >>> idx.is_object()
        False
        """
        # Do not remove until pandas removes this.
        warnings.warn(
            f"{type(self).__name__}.is_object is deprecated. "
            "Use cudf.api.types.is_object_dtype instead",
            FutureWarning,
        )
        return self._is_object()

    def _is_object(self):
        raise NotImplementedError

    def is_categorical(self):
        """
        Check if the Index holds categorical data.

        .. deprecated:: 23.04
           Use `cudf.api.types.is_categorical_dtype` instead.

        Returns
        -------
        bool
            True if the Index is categorical.

        See Also
        --------
        CategoricalIndex : Index for categorical data.
        is_boolean : Check if the Index only consists of booleans.
        is_integer : Check if the Index only consists of integers.
        is_floating : Check if the Index is a floating type.
        is_numeric : Check if the Index only consists of numeric data.
        is_object : Check if the Index is of the object dtype.
        is_interval : Check if the Index holds Interval objects.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index(["Watermelon", "Orange", "Apple",
        ...                 "Watermelon"]).astype("category")
        >>> idx.is_categorical()
        True
        >>> idx = cudf.Index([1, 3, 5, 7])
        >>> idx.is_categorical()
        False
        >>> s = cudf.Series(["Peter", "Victor", "Elisabeth", "Mar"])
        >>> s
        0        Peter
        1       Victor
        2    Elisabeth
        3          Mar
        dtype: object
        >>> s.index.is_categorical()
        False
        """
        # Do not remove until pandas removes this.
        warnings.warn(
            f"{type(self).__name__}.is_categorical is deprecated. "
            "Use cudf.api.types.is_categorical_dtype instead",
            FutureWarning,
        )
        return self._is_categorical()

    def _is_categorical(self):
        raise NotImplementedError

    def is_interval(self):
        """
        Check if the Index holds Interval objects.

        .. deprecated:: 23.04
           Use `cudf.api.types.is_interval_dtype` instead.

        Returns
        -------
        bool
            Whether or not the Index holds Interval objects.

        See Also
        --------
        IntervalIndex : Index for Interval objects.
        is_boolean : Check if the Index only consists of booleans.
        is_integer : Check if the Index only consists of integers.
        is_floating : Check if the Index is a floating type.
        is_numeric : Check if the Index only consists of numeric data.
        is_object : Check if the Index is of the object dtype.
        is_categorical : Check if the Index holds categorical data.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> idx = cudf.from_pandas(
        ...     pd.Index([pd.Interval(left=0, right=5),
        ...               pd.Interval(left=5, right=10)])
        ... )
        >>> idx.is_interval()
        True
        >>> idx = cudf.Index([1, 3, 5, 7])
        >>> idx.is_interval()
        False
        """
        # Do not remove until pandas removes this.
        warnings.warn(
            f"{type(self).__name__}.is_interval is deprecated. "
            "Use cudf.api.types.is_interval_dtype instead",
            FutureWarning,
        )
        return self._is_interval()

    def _is_interval(self):
        raise NotImplementedError

    def _union(self, other, sort=None):
        # TODO: As a future optimization we should explore
        # not doing `to_frame`
        self_df = self.to_frame(index=False, name=0)
        other_df = other.to_frame(index=False, name=0)
        self_df["order"] = self_df.index
        other_df["order"] = other_df.index
        res = self_df.merge(other_df, on=[0], how="outer")
        res = res.sort_values(
            by=res._data.to_pandas_index()[1:], ignore_index=True
        )
        union_result = cudf.core.index._index_from_data({0: res._data[0]})

        if sort in {None, True} and len(other):
            return union_result.sort_values()
        return union_result

    def _intersection(self, other, sort=None):
        intersection_result = cudf.core.index._index_from_data(
            cudf.DataFrame._from_data({"None": self.unique()._column})
            .merge(
                cudf.DataFrame._from_data({"None": other.unique()._column}),
                how="inner",
                on="None",
            )
            ._data
        )

        if sort is {None, True} and len(other):
            return intersection_result.sort_values()
        return intersection_result

    def sort_values(
        self,
        return_indexer=False,
        ascending=True,
        na_position="last",
        key=None,
    ):
        """
        Return a sorted copy of the index, and optionally return the indices
        that sorted the index itself.

        Parameters
        ----------
        return_indexer : bool, default False
            Should the indices that would sort the index be returned.
        ascending : bool, default True
            Should the index values be sorted in an ascending order.
        na_position : {'first' or 'last'}, default 'last'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.
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
        cudf.Series.min : Sort values of a Series.
        cudf.DataFrame.sort_values : Sort values in a DataFrame.

        Examples
        --------
        >>> import cudf
        >>> idx = cudf.Index([10, 100, 1, 1000])
        >>> idx
        Index([10, 100, 1, 1000], dtype='int64')

        Sort values in ascending order (default behavior).

        >>> idx.sort_values()
        Index([1, 10, 100, 1000], dtype='int64')

        Sort values in descending order, and also get the indices `idx` was
        sorted by.

        >>> idx.sort_values(ascending=False, return_indexer=True)
        (Index([1000, 100, 10, 1], dtype='int64'), array([3, 1, 0, 2],
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
        if na_position not in {"first", "last"}:
            raise ValueError(f"invalid na_position: {na_position}")

        indices = self.argsort(ascending=ascending, na_position=na_position)
        index_sorted = self.take(indices)

        if return_indexer:
            return index_sorted, indices
        else:
            return index_sorted

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
        >>> lhs = cudf.DataFrame({
        ...     "a": [2, 3, 1],
        ...     "b": [3, 4, 2],
        ... }).set_index(['a', 'b']).index
        >>> lhs
        MultiIndex([(2, 3),
                    (3, 4),
                    (1, 2)],
                   names=['a', 'b'])
        >>> rhs = cudf.DataFrame({"a": [1, 4, 3]}).set_index('a').index
        >>> rhs
        Index([1, 4, 3], dtype='int64', name='a')
        >>> lhs.join(rhs, how='inner')
        MultiIndex([(3, 4),
                    (1, 2)],
                   names=['a', 'b'])
        """
        if return_indexers is not False:
            raise NotImplementedError("return_indexers is not implemented")
        self_is_multi = isinstance(self, cudf.MultiIndex)
        other_is_multi = isinstance(other, cudf.MultiIndex)
        if level is not None:
            if self_is_multi and other_is_multi:
                raise TypeError(
                    "Join on level between two MultiIndex objects is ambiguous"
                )

            if not is_scalar(level):
                raise ValueError("level should be an int or a label only")

        if other_is_multi:
            if how == "left":
                how = "right"
            elif how == "right":
                how = "left"
            rhs = self.copy(deep=False)
            lhs = other.copy(deep=False)
        else:
            lhs = self.copy(deep=False)
            rhs = other.copy(deep=False)
        same_names = lhs.names == rhs.names
        # There should be no `None` values in Joined indices,
        # so essentially it would be `left/right` or 'inner'
        # in case of MultiIndex
        if isinstance(lhs, cudf.MultiIndex):
            on = (
                lhs._data.select_by_index(level).names[0]
                if isinstance(level, int)
                else level
            )

            if on is not None:
                rhs.names = (on,)
            on = rhs.names[0]
            if how == "outer":
                how = "left"
            elif how == "right":
                how = "inner"
        else:
            # Both are normal indices
            on = lhs.names[0]
            rhs.names = lhs.names

        lhs = lhs.to_frame()
        rhs = rhs.to_frame()

        output = lhs.merge(rhs, how=how, on=on, sort=sort)

        # If both inputs were MultiIndexes, the output is a MultiIndex.
        # Otherwise, the output is only a MultiIndex if there are multiple
        # columns
        if self_is_multi and other_is_multi:
            return cudf.MultiIndex._from_data(output._data)
        else:
            idx = cudf.core.index._index_from_data(output._data)
            idx.name = self.name if same_names else None
            return idx

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
        Index([1, 2, 3], dtype='int64', name='one')
        >>> index.name
        'one'
        >>> renamed_index = index.rename('two')
        >>> renamed_index
        Index([1, 2, 3], dtype='int64', name='two')
        >>> renamed_index.name
        'two'
        """
        if inplace is True:
            self.name = name
            return None
        else:
            out = self.copy(deep=True)
            out.name = name
            return out

    def _indices_of(self, value) -> cudf.core.column.NumericalColumn:
        """
        Return indices corresponding to value

        Parameters
        ----------
        value
            Value to look for in index

        Returns
        -------
        Column of indices
        """
        raise NotImplementedError

    def find_label_range(self, loc: slice) -> slice:
        """
        Translate a label-based slice to an index-based slice

        Parameters
        ----------
        loc
            slice to search for.

        Notes
        -----
        As with all label-based searches, the slice is right-closed.

        Returns
        -------
        New slice translated into integer indices of the index (right-open).
        """
        start = loc.start
        stop = loc.stop
        step = 1 if loc.step is None else loc.step
        start_side: Literal["left", "right"]
        stop_side: Literal["left", "right"]
        if step < 0:
            start_side, stop_side = "right", "left"
        else:
            start_side, stop_side = "left", "right"
        istart = (
            None
            if start is None
            else self.get_slice_bound(start, side=start_side)
        )
        istop = (
            None
            if stop is None
            else self.get_slice_bound(stop, side=stop_side)
        )
        if step < 0:
            # Fencepost
            istart = None if istart is None else max(istart - 1, 0)
            istop = None if (istop is None or istop == 0) else istop - 1
        return slice(istart, istop, step)

    def searchsorted(
        self,
        value,
        side: Literal["left", "right"] = "left",
        ascending: bool = True,
        na_position: Literal["first", "last"] = "last",
    ):
        """Find index where elements should be inserted to maintain order

        Parameters
        ----------
        value :
            Value to be hypothetically inserted into Self
        side : str {'left', 'right'} optional, default 'left'
            If 'left', the index of the first suitable location found is given
            If 'right', return the last such index
        ascending : bool optional, default True
            Index is in ascending order (otherwise descending)
        na_position : str {'last', 'first'} optional, default 'last'
            Position of null values in sorted order

        Returns
        -------
        Insertion point.

        Notes
        -----
        As a precondition the index must be sorted in the same order
        as requested by the `ascending` flag.
        """
        raise NotImplementedError

    def get_slice_bound(
        self,
        label,
        side: Literal["left", "right"],
    ) -> int:
        """
        Calculate slice bound that corresponds to given label.
        Returns leftmost (one-past-the-rightmost if ``side=='right'``) position
        of given label.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}

        Returns
        -------
        int
            Index of label.
        """
        if side not in {"left", "right"}:
            raise ValueError(f"Invalid side argument {side}")
        if self.is_monotonic_increasing or self.is_monotonic_decreasing:
            return self.searchsorted(
                label, side=side, ascending=self.is_monotonic_increasing
            )
        else:
            try:
                left, right = self._values._find_first_and_last(label)
            except ValueError:
                raise KeyError(f"{label=} not in index")
            if left != right:
                raise KeyError(
                    f"Cannot get slice bound for non-unique label {label=}"
                )
            if side == "left":
                return left
            else:
                return right + 1

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

        handled_types = [BaseIndex, cudf.Series]

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
                result = cudf_func(*args, **kwargs)
                if fname == "unique":
                    # NumPy expects a sorted result for `unique`, which is not
                    # guaranteed by cudf.Index.unique.
                    result = result.sort_values()
                return result

        else:
            return NotImplemented

    @classmethod
    def from_pandas(cls, index: pd.Index, nan_as_null=no_default):
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
        >>> cudf.Index.from_pandas(pdi)
        Index([10.0, 20.0, 30.0, <NA>], dtype='float64')
        >>> cudf.Index.from_pandas(pdi, nan_as_null=False)
        Index([10.0, 20.0, 30.0, nan], dtype='float64')
        """
        if nan_as_null is no_default:
            nan_as_null = (
                False if cudf.get_option("mode.pandas_compatible") else None
            )

        if not isinstance(index, pd.Index):
            raise TypeError("not a pandas.Index")
        if isinstance(index, pd.RangeIndex):
            return cudf.RangeIndex(
                start=index.start,
                stop=index.stop,
                step=index.step,
                name=index.name,
            )
        else:
            return cudf.Index(
                column.as_column(index, nan_as_null=nan_as_null),
                name=index.name,
            )

    @property
    def _constructor_expanddim(self):
        return cudf.MultiIndex

    def drop_duplicates(
        self,
        keep="first",
        nulls_are_equal=True,
    ):
        """
        Drop duplicate rows in index.

        keep : {"first", "last", False}, default "first"
            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.
        nulls_are_equal: bool, default True
            Null elements are considered equal to other null elements.
        """

        # This utilizes the fact that all `Index` is also a `Frame`.
        # Except RangeIndex.
        return self._from_columns_like_self(
            drop_duplicates(
                list(self._columns),
                keys=range(len(self._data)),
                keep=keep,
                nulls_are_equal=nulls_are_equal,
            ),
            self._column_names,
        )

    def duplicated(self, keep="first"):
        """
        Indicate duplicate index values.

        Duplicated values are indicated as ``True`` values in the resulting
        array. Either all duplicates, all except the first, or all except the
        last occurrence of duplicates can be indicated.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            The value or values in a set of duplicates to mark as missing.

            - ``'first'`` : Mark duplicates as ``True`` except for the first
              occurrence.
            - ``'last'`` : Mark duplicates as ``True`` except for the last
              occurrence.
            - ``False`` : Mark all duplicates as ``True``.

        Returns
        -------
        cupy.ndarray[bool]

        See Also
        --------
        Series.duplicated : Equivalent method on cudf.Series.
        DataFrame.duplicated : Equivalent method on cudf.DataFrame.
        Index.drop_duplicates : Remove duplicate values from Index.

        Examples
        --------
        By default, for each set of duplicated values, the first occurrence is
        set to False and all others to True:

        >>> import cudf
        >>> idx = cudf.Index(['lama', 'cow', 'lama', 'beetle', 'lama'])
        >>> idx.duplicated()
        array([False, False,  True, False,  True])

        which is equivalent to

        >>> idx.duplicated(keep='first')
        array([False, False,  True, False,  True])

        By using 'last', the last occurrence of each set of duplicated values
        is set to False and all others to True:

        >>> idx.duplicated(keep='last')
        array([ True, False,  True, False, False])

        By setting keep to ``False``, all duplicates are True:

        >>> idx.duplicated(keep=False)
        array([ True, False,  True, False,  True])
        """
        return self.to_series().duplicated(keep=keep).to_cupy()

    def dropna(self, how="any"):
        """
        Drop null rows from Index.

        how : {"any", "all"}, default "any"
            Specifies how to decide whether to drop a row.
            "any" (default) drops rows containing at least
            one null value. "all" drops only rows containing
            *all* null values.
        """
        if how not in {"any", "all"}:
            raise ValueError(f"{how=} must be 'any' or 'all'")
        try:
            if not self.hasnans:
                return self.copy()
        except NotImplementedError:
            pass
        # This is to be consistent with IndexedFrame.dropna to handle nans
        # as nulls by default
        data_columns = [
            col.nans_to_nulls()
            if isinstance(col, cudf.core.column.NumericalColumn)
            else col
            for col in self._columns
        ]

        return self._from_columns_like_self(
            drop_nulls(
                data_columns,
                how=how,
                keys=range(len(data_columns)),
            ),
            self._column_names,
        )

    def _gather(self, gather_map, nullify=False, check_bounds=True):
        """Gather rows of index specified by indices in `gather_map`.

        Skip bounds checking if check_bounds is False.
        Set rows to null for all out of bound indices if nullify is `True`.
        """
        gather_map = cudf.core.column.as_column(gather_map)

        # TODO: For performance, the check and conversion of gather map should
        # be done by the caller. This check will be removed in future release.
        if not is_integer_dtype(gather_map.dtype):
            gather_map = gather_map.astype(size_type_dtype)

        if not _gather_map_is_valid(
            gather_map, len(self), check_bounds, nullify
        ):
            raise IndexError("Gather map index is out of bounds.")

        return self._from_columns_like_self(
            gather(list(self._columns), gather_map, nullify=nullify),
            self._column_names,
        )

    def take(self, indices, axis=0, allow_fill=True, fill_value=None):
        """Return a new index containing the rows specified by *indices*

        Parameters
        ----------
        indices : array-like
            Array of ints indicating which positions to take.
        axis : int
            The axis over which to select values, always 0.
        allow_fill : Unsupported
        fill_value : Unsupported

        Returns
        -------
        out : Index
            New object with desired subset of rows.

        Examples
        --------
        >>> idx = cudf.Index(['a', 'b', 'c', 'd', 'e'])
        >>> idx.take([2, 0, 4, 3])
        Index(['c', 'a', 'e', 'd'], dtype='object')
        """

        if axis not in {0, "index"}:
            raise NotImplementedError(
                "Gather along column axis is not yet supported."
            )
        if not allow_fill or fill_value is not None:
            raise NotImplementedError(
                "`allow_fill` and `fill_value` are unsupported."
            )

        return self._gather(indices)

    def _apply_boolean_mask(self, boolean_mask):
        """Apply boolean mask to each row of `self`.

        Rows corresponding to `False` is dropped.
        """
        boolean_mask = cudf.core.column.as_column(boolean_mask)
        if not is_bool_dtype(boolean_mask.dtype):
            raise ValueError("boolean_mask is not boolean type.")

        return self._from_columns_like_self(
            apply_boolean_mask(list(self._columns), boolean_mask),
            column_names=self._column_names,
        )

    def repeat(self, repeats, axis=None):
        """Repeat elements of a Index.

        Returns a new Index where each element of the current Index is repeated
        consecutively a given number of times.

        Parameters
        ----------
        repeats : int, or array of ints
            The number of repetitions for each element. This should
            be a non-negative integer. Repeating 0 times will return
            an empty object.

        Returns
        -------
        Index
            A newly created object of same type as caller with repeated
            elements.

        Examples
        --------
        >>> index = cudf.Index([10, 22, 33, 55])
        >>> index
        Index([10, 22, 33, 55], dtype='int64')
        >>> index.repeat(5)
        Index([10, 10, 10, 10, 10, 22, 22, 22, 22, 22, 33,
                    33, 33, 33, 33, 55, 55, 55, 55, 55],
                dtype='int64')
        """
        raise NotImplementedError

    def _new_index_for_reset_index(
        self, levels: tuple | None, name
    ) -> None | BaseIndex:
        """Return the new index after .reset_index"""
        # None is caught later to return RangeIndex
        return None

    def _columns_for_reset_index(
        self, levels: tuple | None
    ) -> Generator[tuple[Any, ColumnBase], None, None]:
        """Return the columns and column names for .reset_index"""
        yield (
            "index" if self.name is None else self.name,
            next(iter(self._columns)),
        )

    def _split(self, splits):
        raise NotImplementedError


def _get_result_name(left_name, right_name):
    return left_name if _is_same_name(left_name, right_name) else None


def _return_get_indexer_result(result):
    if cudf.get_option("mode.pandas_compatible"):
        return result.astype("int64")
    return result
