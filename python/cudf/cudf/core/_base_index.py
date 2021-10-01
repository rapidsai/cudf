# Copyright (c) 2021, NVIDIA CORPORATION.

from __future__ import annotations, division, print_function

import pickle
from typing import Any, Set

import cupy
import pandas as pd

import cudf
from cudf._typing import DtypeObj
from cudf.api.types import is_dtype_equal, is_integer, is_list_like, is_scalar
from cudf.core.abc import Serializable
from cudf.core.column import ColumnBase, column
from cudf.core.column_accessor import ColumnAccessor
from cudf.utils import ioutils
from cudf.utils.dtypes import (
    is_mixed_with_object_dtype,
    numeric_normalize_types,
)
from cudf.utils.utils import cached_property


class BaseIndex(Serializable):
    """Base class for all cudf Index types."""

    dtype: DtypeObj
    _accessors: Set[Any] = set()
    _data: ColumnAccessor

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        if method == "__call__" and hasattr(cudf, ufunc.__name__):
            func = getattr(cudf, ufunc.__name__)
            return func(*inputs)
        else:
            return NotImplemented

    @cached_property
    def _values(self) -> ColumnBase:
        raise NotImplementedError

    def copy(self, deep: bool = True) -> BaseIndex:
        raise NotImplementedError

    @property
    def size(self):
        # The size of an index is always its length irrespective of dimension.
        return len(self)

    @property
    def values(self):
        return self._values.values

    def get_loc(self, key, method=None, tolerance=None):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError()

    def __contains__(self, item):
        return item in self._values

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
    def is_monotonic(self):
        """Return boolean if values in the object are monotonic_increasing.

        This property is an alias for :attr:`is_monotonic_increasing`.

        Returns
        -------
        bool
        """
        return self.is_monotonic_increasing

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

            other = cudf.Index(other)

            if len(self) == 0:
                to_concat = [other]

            if len(self) and len(other):
                if is_mixed_with_object_dtype(this, other):
                    got_dtype = (
                        other.dtype
                        if this.dtype == cudf.dtype("object")
                        else this.dtype
                    )
                    raise TypeError(
                        f"cudf does not support appending an Index of "
                        f"dtype `{cudf.dtype('object')}` with an Index "
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

        other = cudf.Index(other)

        if is_mixed_with_object_dtype(self, other):
            difference = self.copy()
        else:
            difference = self.join(other, how="leftanti")
            if self.dtype != other.dtype:
                difference = difference.astype(self.dtype)

        if sort is None and len(other):
            return difference.sort_values()

        return difference

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
        cudf.Series.min : Sort values of a Series.
        cudf.DataFrame.sort_values : Sort values in a DataFrame.

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
        index_sorted = cudf.Index(self.take(indices), name=self.name)

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
        return cudf.Index(self._values.unique(), name=self.name)

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

        return cudf.Index(
            self.copy(deep=copy)._values.astype(dtype), name=self.name
        )

    # TODO: This method is deprecated and can be removed.
    def to_array(self, fillna=None):
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

    def get_slice_bound(self, label, side, kind=None):
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
        >>> cudf.Index.from_pandas(pdi)
        Float64Index([10.0, 20.0, 30.0, <NA>], dtype='float64')
        >>> cudf.Index.from_pandas(pdi, nan_as_null=False)
        Float64Index([10.0, 20.0, 30.0, nan], dtype='float64')
        """
        if not isinstance(index, pd.Index):
            raise TypeError("not a pandas.Index")

        ind = cudf.Index(column.as_column(index, nan_as_null=nan_as_null))
        ind.name = index.name
        return ind

    @property
    def _constructor_expanddim(self):
        return cudf.MultiIndex
