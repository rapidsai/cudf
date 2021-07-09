# Copyright (c) 2020-2021, NVIDIA CORPORATION.

from __future__ import annotations

import copy
import functools
import warnings
from collections import abc
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, TypeVar, Union

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
from nvtx import annotate

import cudf
from cudf import _lib as libcudf
from cudf._typing import ColumnLike, DataFrameOrSeries
from cudf.api.types import is_dict_like, is_dtype_equal
from cudf.core.column import (
    ColumnBase,
    as_column,
    build_categorical_column,
    column_empty,
)
from cudf.core.join import merge
from cudf.utils.dtypes import (
    _is_non_decimal_numeric_dtype,
    find_common_type,
    is_categorical_dtype,
    is_column_like,
    is_decimal_dtype,
    is_integer_dtype,
    is_numerical_dtype,
    is_scalar,
    min_scalar_type,
)

T = TypeVar("T", bound="Frame")

if TYPE_CHECKING:
    from cudf.core.column_accessor import ColumnAccessor


class Frame(libcudf.table.Table):
    """
    Frame: A collection of Column objects with an optional index.

    Parameters
    ----------
    data : dict
        An dict mapping column names to Columns
    index : Table
        A Frame representing the (optional) index columns.
    """

    _data: "ColumnAccessor"

    @classmethod
    def __init_subclass__(cls):
        # All subclasses contain a set _accessors that is used to hold custom
        # accessors defined by user APIs (see cudf/api/extensions/accessor.py).
        cls._accessors = set()

    @classmethod
    def _from_table(cls, table: Frame):
        return cls(table._data, index=table._index)

    def _mimic_inplace(
        self: T, result: Frame, inplace: bool = False
    ) -> Optional[Frame]:
        if inplace:
            for col in self._data:
                if col in result._data:
                    self._data[col]._mimic_inplace(
                        result._data[col], inplace=True
                    )
            self._data = result._data
            self._index = result._index
            return None
        else:
            return result

    @property
    def size(self):
        """
        Return the number of elements in the underlying data.

        Returns
        -------
        size : Size of the DataFrame / Index / Series / MultiIndex

        Examples
        --------
        Size of an empty dataframe is 0.

        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df
        Empty DataFrame
        Columns: []
        Index: []
        >>> df.size
        0
        >>> df = cudf.DataFrame(index=[1, 2, 3])
        >>> df
        Empty DataFrame
        Columns: []
        Index: [1, 2, 3]
        >>> df.size
        0

        DataFrame with values

        >>> df = cudf.DataFrame({'a': [10, 11, 12],
        ...         'b': ['hello', 'rapids', 'ai']})
        >>> df
            a       b
        0  10   hello
        1  11  rapids
        2  12      ai
        >>> df.size
        6
        >>> df.index
        RangeIndex(start=0, stop=3)
        >>> df.index.size
        3

        Size of an Index

        >>> index = cudf.Index([])
        >>> index
        Float64Index([], dtype='float64')
        >>> index.size
        0
        >>> index = cudf.Index([1, 2, 3, 10])
        >>> index
        Int64Index([1, 2, 3, 10], dtype='int64')
        >>> index.size
        4

        Size of a MultiIndex

        >>> midx = cudf.MultiIndex(
        ...                 levels=[["a", "b", "c", None], ["1", None, "5"]],
        ...                 codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
        ...                 names=["x", "y"],
        ...             )
        >>> midx
        MultiIndex([( 'a',  '1'),
                    ( 'a',  '5'),
                    ( 'b', <NA>),
                    ( 'c', <NA>),
                    (<NA>,  '1')],
                   names=['x', 'y'])
        >>> midx.size
        5
        """
        return self._num_columns * self._num_rows

    @property
    def _is_homogeneous(self):
        # make sure that the dataframe has columns
        if not self._data.columns:
            return True

        first_type = self._data.columns[0].dtype.name
        return all(x.dtype.name == first_type for x in self._data.columns)

    @property
    def empty(self):
        """
        Indicator whether DataFrame or Series is empty.

        True if DataFrame/Series is entirely empty (no items),
        meaning any of the axes are of length 0.

        Returns
        -------
        out : bool
            If DataFrame/Series is empty, return True, if not return False.

        Notes
        -----
        If DataFrame/Series contains only `null` values, it is still not
        considered empty. See the example below.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'A' : []})
        >>> df
        Empty DataFrame
        Columns: [A]
        Index: []
        >>> df.empty
        True

        If we only have `null` values in our DataFrame, it is
        not considered empty! We will need to drop
        the `null`'s to make the DataFrame empty:

        >>> df = cudf.DataFrame({'A' : [None, None]})
        >>> df
              A
        0  <NA>
        1  <NA>
        >>> df.empty
        False
        >>> df.dropna().empty
        True

        Non-empty and empty Series example:

        >>> s = cudf.Series([1, 2, None])
        >>> s
        0       1
        1       2
        2    <NA>
        dtype: int64
        >>> s.empty
        False
        >>> s = cudf.Series([])
        >>> s
        Series([], dtype: float64)
        >>> s.empty
        True
        """
        return self.size == 0

    def __len__(self):
        return self._num_rows

    def copy(self: T, deep: bool = True) -> T:
        """
        Make a copy of this object's indices and data.

        When ``deep=True`` (default), a new object will be created with a
        copy of the calling object's data and indices. Modifications to
        the data or indices of the copy will not be reflected in the
        original object (see notes below).
        When ``deep=False``, a new object will be created without copying
        the calling object's data or index (only references to the data
        and index are copied). Any changes to the data of the original
        will be reflected in the shallow copy (and vice versa).

        Parameters
        ----------
        deep : bool, default True
            Make a deep copy, including a copy of the data and the indices.
            With ``deep=False`` neither the indices nor the data are copied.

        Returns
        -------
        copy : Series or DataFrame
            Object type matches caller.

        Examples
        --------
        >>> s = cudf.Series([1, 2], index=["a", "b"])
        >>> s
        a    1
        b    2
        dtype: int64
        >>> s_copy = s.copy()
        >>> s_copy
        a    1
        b    2
        dtype: int64

        **Shallow copy versus default (deep) copy:**

        >>> s = cudf.Series([1, 2], index=["a", "b"])
        >>> deep = s.copy()
        >>> shallow = s.copy(deep=False)

        Shallow copy shares data and index with original.

        >>> s is shallow
        False
        >>> s._column is shallow._column and s.index is shallow.index
        True

        Deep copy has own copy of data and index.

        >>> s is deep
        False
        >>> s.values is deep.values or s.index is deep.index
        False

        Updates to the data shared by shallow copy and original is reflected
        in both; deep copy remains unchanged.

        >>> s['a'] = 3
        >>> shallow['b'] = 4
        >>> s
        a    3
        b    4
        dtype: int64
        >>> shallow
        a    3
        b    4
        dtype: int64
        >>> deep
        a    1
        b    2
        dtype: int64
        """
        new_frame = self.__class__.__new__(type(self))
        new_frame._data = self._data.copy(deep=deep)

        if self._index is not None:
            new_frame._index = self._index.copy(deep=deep)
        else:
            new_frame._index = None

        return new_frame

    @classmethod
    @annotate("CONCAT", color="orange", domain="cudf_python")
    def _concat(
        cls, objs, axis=0, join="outer", ignore_index=False, sort=False
    ):
        # flag to indicate at least one empty input frame also has an index
        empty_has_index = False
        # length of output frame's RangeIndex if all input frames are empty,
        # and at least one has an index
        result_index_length = 0
        # the number of empty input frames
        num_empty_input_frames = 0

        for i, obj in enumerate(objs):
            # shallow-copy the input DFs in case the same DF instance
            # is concatenated with itself
            objs[i] = obj.copy(deep=False)

            # If ignore_index is true, determine if
            # all or some objs are empty(and have index).
            # 1. If all objects are empty(and have index), we
            # should set the index separately using RangeIndex.
            # 2. If some objects are empty(and have index), we
            # create empty columns later while populating `columns`
            # variable. Detailed explanation of second case before
            # allocation of `columns` variable below.
            if ignore_index and obj.empty:
                num_empty_input_frames += 1
                result_index_length += len(obj)
                empty_has_index = empty_has_index or len(obj) > 0

        if join == "inner":
            sets_of_column_names = [set(obj._column_names) for obj in objs]

            intersecting_columns = functools.reduce(
                set.intersection, sets_of_column_names
            )
            union_of_columns = functools.reduce(
                set.union, sets_of_column_names
            )
            non_intersecting_columns = union_of_columns.symmetric_difference(
                intersecting_columns
            )

            # Get an ordered list of the intersecting columns to preserve input
            # order, which is promised by pandas for inner joins.
            ordered_intersecting_columns = [
                name
                for obj in objs
                for name in obj._column_names
                if name in intersecting_columns
            ]

            names = dict.fromkeys(ordered_intersecting_columns).keys()

            if axis == 0:
                if ignore_index and (
                    num_empty_input_frames > 0
                    or len(intersecting_columns) == 0
                ):
                    # When ignore_index is True and if there is
                    # at least 1 empty dataframe and no
                    # intersecting columns are present, an empty dataframe
                    # needs to be returned just with an Index.
                    empty_has_index = True
                    num_empty_input_frames = len(objs)
                    result_index_length = sum(len(obj) for obj in objs)

                # remove columns not present in all objs
                for obj in objs:
                    obj.drop(
                        columns=non_intersecting_columns,
                        inplace=True,
                        errors="ignore",
                    )
        elif join == "outer":
            # Get a list of the unique table column names
            names = [name for f in objs for name in f._column_names]
            names = dict.fromkeys(names).keys()

        else:
            raise ValueError(
                "Only can inner (intersect) or outer (union) when joining"
                "the other axis"
            )

        if sort:
            try:
                # Sorted always returns a list, but will fail to sort if names
                # include different types that are not comparable.
                names = sorted(names)
            except TypeError:
                names = list(names)
        else:
            names = list(names)

        # Combine the index and table columns for each Frame into a list of
        # [...index_cols, ...table_cols].
        #
        # If any of the input frames have a non-empty index, include these
        # columns in the list of columns to concatenate, even if the input
        # frames are empty and `ignore_index=True`.
        columns = [
            (
                []
                if (ignore_index and not empty_has_index)
                else list(f._index._data.columns)
            )
            + [f._data[name] if name in f._data else None for name in names]
            for f in objs
        ]

        # Get a list of the combined index and table column indices
        indices = list(range(functools.reduce(max, map(len, columns))))
        # The position of the first table colum in each
        # combined index + table columns list
        first_data_column_position = len(indices) - len(names)

        # Get the non-null columns and their dtypes
        non_null_cols, dtypes = _get_non_null_cols_and_dtypes(indices, columns)

        # Infer common dtypes between numeric columns
        # and combine CategoricalColumn categories
        categories = _find_common_dtypes_and_categories(non_null_cols, dtypes)

        # Cast all columns to a common dtype, assign combined categories,
        # and back-fill missing columns with all-null columns
        _cast_cols_to_common_dtypes(indices, columns, dtypes, categories)

        # Construct input tables with the index and data columns in the same
        # order. This strips the given index/column names and replaces the
        # names with their integer positions in the `cols` list
        tables = []
        for cols in columns:
            table_index = None
            if 1 == first_data_column_position:
                table_index = cudf.core.index.as_index(cols[0])
            elif first_data_column_position > 1:
                table_index = libcudf.table.Table(
                    data=dict(
                        zip(
                            indices[:first_data_column_position],
                            cols[:first_data_column_position],
                        )
                    )
                )
            tables.append(
                libcudf.table.Table(
                    data=dict(
                        zip(
                            indices[first_data_column_position:],
                            cols[first_data_column_position:],
                        )
                    ),
                    index=table_index,
                )
            )

        # Concatenate the Tables
        out = cls._from_table(
            libcudf.concat.concat_tables(tables, ignore_index=ignore_index)
        )

        # If ignore_index is True, all input frames are empty, and at
        # least one input frame has an index, assign a new RangeIndex
        # to the result frame.
        if empty_has_index and num_empty_input_frames == len(objs):
            out._index = cudf.RangeIndex(result_index_length)
        # Reassign the categories for any categorical table cols
        _reassign_categories(
            categories, out._data, indices[first_data_column_position:]
        )

        # Reassign the categories for any categorical index cols
        if not isinstance(out._index, cudf.RangeIndex):
            _reassign_categories(
                categories,
                out._index._data,
                indices[:first_data_column_position],
            )
            if not isinstance(
                out._index, cudf.MultiIndex
            ) and is_categorical_dtype(out._index._values.dtype):
                out = out.set_index(
                    cudf.core.index.as_index(out.index._values)
                )

        # Reassign precision for any decimal cols
        for name, col in out._data.items():
            if isinstance(col, cudf.core.column.Decimal64Column):
                col = col._with_type_metadata(tables[0]._data[name].dtype)

        # Reassign index and column names
        if isinstance(objs[0].columns, pd.MultiIndex):
            out.columns = objs[0].columns
        else:
            out.columns = names
        if not ignore_index:
            out._index.name = objs[0]._index.name
            out._index.names = objs[0]._index.names

        return out

    def equals(self, other, **kwargs):
        """
        Test whether two objects contain the same elements.
        This function allows two Series or DataFrames to be compared against
        each other to see if they have the same shape and elements. NaNs in
        the same location are considered equal. The column headers do not
        need to have the same type.

        Parameters
        ----------
        other : Series or DataFrame
            The other Series or DataFrame to be compared with the first.

        Returns
        -------
        bool
            True if all elements are the same in both objects, False
            otherwise.

        Examples
        --------
        >>> import cudf

        Comparing Series with `equals`:

        >>> s = cudf.Series([1, 2, 3])
        >>> other = cudf.Series([1, 2, 3])
        >>> s.equals(other)
        True
        >>> different = cudf.Series([1.5, 2, 3])
        >>> s.equals(different)
        False

        Comparing DataFrames with `equals`:

        >>> df = cudf.DataFrame({1: [10], 2: [20]})
        >>> df
            1   2
        0  10  20
        >>> exactly_equal = cudf.DataFrame({1: [10], 2: [20]})
        >>> exactly_equal
            1   2
        0  10  20
        >>> df.equals(exactly_equal)
        True

        For two DataFrames to compare equal, the types of column
        values must be equal, but the types of column labels
        need not:

        >>> different_column_type = cudf.DataFrame({1.0: [10], 2.0: [20]})
        >>> different_column_type
           1.0  2.0
        0   10   20
        >>> df.equals(different_column_type)
        True
        """
        if self is other:
            return True

        check_types = kwargs.get("check_types", True)

        if check_types:
            if type(self) is not type(other):
                return False

        if other is None or len(self) != len(other):
            return False

        # check data:
        for self_col, other_col in zip(
            self._data.values(), other._data.values()
        ):
            if not self_col.equals(other_col, check_dtypes=check_types):
                return False

        # check index:
        if self._index is None:
            return other._index is None
        else:
            return self._index.equals(other._index)

    def _explode(self, explode_column: Any, ignore_index: bool):
        """Helper function for `explode` in `Series` and `Dataframe`, explodes
        a specified nested column. Other columns' corresponding rows are
        duplicated. If ignore_index is set, the original index is not exploded
        and will be replaced with a `RangeIndex`.
        """
        explode_column_num = self._column_names.index(explode_column)
        if not ignore_index and self._index is not None:
            explode_column_num += self._index.nlevels

        res_tbl = libcudf.lists.explode_outer(
            self, explode_column_num, ignore_index
        )
        res = self.__class__._from_table(res_tbl)

        res._data.multiindex = self._data.multiindex
        res._data._level_names = self._data._level_names

        if not ignore_index and self._index is not None:
            res.index.names = self._index.names
        return res

    def _get_columns_by_label(self, labels, downcast=False):
        """
        Returns columns of the Frame specified by `labels`

        """
        return self._data.select_by_label(labels)

    def _get_columns_by_index(self, indices):
        """
        Returns columns of the Frame specified by `labels`

        """
        data = self._data.select_by_index(indices)
        return self.__class__(
            data, columns=data.to_pandas_index(), index=self.index
        )

    def _gather(self, gather_map, keep_index=True, nullify=False):
        if not is_integer_dtype(gather_map.dtype):
            gather_map = gather_map.astype("int32")
        result = self.__class__._from_table(
            libcudf.copying.gather(
                self,
                as_column(gather_map),
                keep_index=keep_index,
                nullify=nullify,
            )
        )
        result._copy_type_metadata(self, include_index=keep_index)
        if keep_index and self._index is not None:
            result._index.names = self._index.names
        return result

    def _hash(self, initial_hash_values=None):
        return libcudf.hash.hash(self, initial_hash_values)

    def _hash_partition(
        self, columns_to_hash, num_partitions, keep_index=True
    ):
        output, offsets = libcudf.hash.hash_partition(
            self, columns_to_hash, num_partitions, keep_index
        )
        output = self.__class__._from_table(output)
        output._copy_type_metadata(self, include_index=keep_index)
        return output, offsets

    def _as_column(self):
        """
        _as_column : Converts a single columned Frame to Column
        """
        assert (
            self._num_columns == 1
            and self._index is None
            and self._column_names[0] is None
        ), """There should be only one data column,
            no index and None as the name to use this method"""

        return self._data[None].copy(deep=False)

    def _scatter(self, key, value):
        result = self._from_table(libcudf.copying.scatter(value, key, self))

        result._copy_type_metadata(self)
        return result

    def _empty_like(self, keep_index=True):
        result = self._from_table(
            libcudf.copying.table_empty_like(self, keep_index)
        )

        result._copy_type_metadata(self, include_index=keep_index)
        return result

    def clip(self, lower=None, upper=None, inplace=False, axis=1):
        """
        Trim values at input threshold(s).

        Assigns values outside boundary to boundary values.
        Thresholds can be singular values or array like,
        and in the latter case the clipping is performed
        element-wise in the specified axis. Currently only
        `axis=1` is supported.

        Parameters
        ----------
        lower : scalar or array_like, default None
            Minimum threshold value. All values below this
            threshold will be set to it. If it is None,
            there will be no clipping based on lower.
            In case of Series/Index, lower is expected to be
            a scalar or an array of size 1.
        upper : scalar or array_like, default None
            Maximum threshold value. All values below this
            threshold will be set to it. If it is None,
            there will be no clipping based on upper.
            In case of Series, upper is expected to be
            a scalar or an array of size 1.
        inplace : bool, default False

        Returns
        -------
        Clipped DataFrame/Series/Index/MultiIndex

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"a":[1, 2, 3, 4], "b":['a', 'b', 'c', 'd']})
        >>> df.clip(lower=[2, 'b'], upper=[3, 'c'])
           a  b
        0  2  b
        1  2  b
        2  3  c
        3  3  c

        >>> df.clip(lower=None, upper=[3, 'c'])
           a  b
        0  1  a
        1  2  b
        2  3  c
        3  3  c

        >>> df.clip(lower=[2, 'b'], upper=None)
           a  b
        0  2  b
        1  2  b
        2  3  c
        3  4  d

        >>> df.clip(lower=2, upper=3, inplace=True)
        >>> df
           a  b
        0  2  2
        1  2  3
        2  3  3
        3  3  3

        >>> import cudf
        >>> sr = cudf.Series([1, 2, 3, 4])
        >>> sr.clip(lower=2, upper=3)
        0    2
        1    2
        2    3
        3    3
        dtype: int64

        >>> sr.clip(lower=None, upper=3)
        0    1
        1    2
        2    3
        3    3
        dtype: int64

        >>> sr.clip(lower=2, upper=None, inplace=True)
        >>> sr
        0    2
        1    2
        2    3
        3    4
        dtype: int64
        """

        if axis != 1:
            raise NotImplementedError("`axis is not yet supported in clip`")

        if lower is None and upper is None:
            return None if inplace is True else self.copy(deep=True)

        if is_scalar(lower):
            lower = np.full(self._num_columns, lower)
        if is_scalar(upper):
            upper = np.full(self._num_columns, upper)

        if len(lower) != len(upper):
            raise ValueError("Length of lower and upper should be equal")

        if len(lower) != self._num_columns:
            raise ValueError(
                """Length of lower/upper should be
                equal to number of columns in
                DataFrame/Series/Index/MultiIndex"""
            )

        output = self.copy(deep=False)
        if output.ndim == 1:
            # In case of series and Index,
            # swap lower and upper if lower > upper
            if (
                lower[0] is not None
                and upper[0] is not None
                and (lower[0] > upper[0])
            ):
                lower[0], upper[0] = upper[0], lower[0]

        for i, name in enumerate(self._data):
            output._data[name] = self._data[name].clip(lower[i], upper[i])

        output._copy_type_metadata(self, include_index=False)

        return self._mimic_inplace(output, inplace=inplace)

    def where(self, cond, other=None, inplace=False):
        """
        Replace values where the condition is False.

        Parameters
        ----------
        cond : bool Series/DataFrame, array-like
            Where cond is True, keep the original value.
            Where False, replace with corresponding value from other.
            Callables are not supported.
        other: scalar, list of scalars, Series/DataFrame
            Entries where cond is False are replaced with
            corresponding value from other. Callables are not
            supported. Default is None.

            DataFrame expects only Scalar or array like with scalars or
            dataframe with same dimension as self.

            Series expects only scalar or series like with same length
        inplace : bool, default False
            Whether to perform the operation in place on the data.

        Returns
        -------
        Same type as caller

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"A":[1, 4, 5], "B":[3, 5, 8]})
        >>> df.where(df % 2 == 0, [-1, -1])
           A  B
        0 -1 -1
        1  4 -1
        2 -1  8

        >>> ser = cudf.Series([4, 3, 2, 1, 0])
        >>> ser.where(ser > 2, 10)
        0     4
        1     3
        2    10
        3    10
        4    10
        dtype: int64
        >>> ser.where(ser > 2)
        0       4
        1       3
        2    <NA>
        3    <NA>
        4    <NA>
        dtype: int64
        """

        return cudf.core._internals.where(
            frame=self, cond=cond, other=other, inplace=inplace
        )

    def mask(self, cond, other=None, inplace=False):
        """
        Replace values where the condition is True.

        Parameters
        ----------
        cond : bool Series/DataFrame, array-like
            Where cond is False, keep the original value.
            Where True, replace with corresponding value from other.
            Callables are not supported.
        other: scalar, list of scalars, Series/DataFrame
            Entries where cond is True are replaced with
            corresponding value from other. Callables are not
            supported. Default is None.

            DataFrame expects only Scalar or array like with scalars or
            dataframe with same dimension as self.

            Series expects only scalar or series like with same length
        inplace : bool, default False
            Whether to perform the operation in place on the data.

        Returns
        -------
        Same type as caller

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"A":[1, 4, 5], "B":[3, 5, 8]})
        >>> df.mask(df % 2 == 0, [-1, -1])
           A  B
        0  1  3
        1 -1  5
        2  5 -1

        >>> ser = cudf.Series([4, 3, 2, 1, 0])
        >>> ser.mask(ser > 2, 10)
        0    10
        1    10
        2     2
        3     1
        4     0
        dtype: int64
        >>> ser.mask(ser > 2)
        0    <NA>
        1    <NA>
        2       2
        3       1
        4       0
        dtype: int64
        """

        if not hasattr(cond, "__invert__"):
            # We Invert `cond` below and call `where`, so
            # making sure the object supports
            # `~`(inversion) operator or `__invert__` method
            cond = cupy.asarray(cond)

        return self.where(cond=~cond, other=other, inplace=inplace)

    def _partition(self, scatter_map, npartitions, keep_index=True):

        output_table, output_offsets = libcudf.partitioning.partition(
            self, scatter_map, npartitions, keep_index
        )
        partitioned = self.__class__._from_table(output_table)

        # due to the split limitation mentioned
        # here: https://github.com/rapidsai/cudf/issues/4607
        # we need to remove first & last elements in offsets.
        # TODO: Remove this after the above issue is fixed.
        output_offsets = output_offsets[1:-1]

        result = partitioned._split(output_offsets, keep_index=keep_index)

        for frame in result:
            frame._copy_type_metadata(self, include_index=keep_index)

        if npartitions:
            for _ in range(npartitions - len(result)):
                result.append(self._empty_like(keep_index))

        return result

    def pipe(self, func, *args, **kwargs):
        """
        Apply ``func(self, *args, **kwargs)``.

        Parameters
        ----------
        func : function
            Function to apply to the Series/DataFrame/Index.
            ``args``, and ``kwargs`` are passed into ``func``.
            Alternatively a ``(callable, data_keyword)`` tuple where
            ``data_keyword`` is a string indicating the keyword of
            ``callable`` that expects the Series/DataFrame/Index.
        args : iterable, optional
            Positional arguments passed into ``func``.
        kwargs : mapping, optional
            A dictionary of keyword arguments passed into ``func``.

        Returns
        -------
        object : the return type of ``func``.

        Examples
        --------

        Use ``.pipe`` when chaining together functions that expect
        Series, DataFrames or GroupBy objects. Instead of writing

        >>> func(g(h(df), arg1=a), arg2=b, arg3=c)

        You can write

        >>> (df.pipe(h)
        ...    .pipe(g, arg1=a)
        ...    .pipe(func, arg2=b, arg3=c)
        ... )

        If you have a function that takes the data as (say) the second
        argument, pass a tuple indicating which keyword expects the
        data. For example, suppose ``f`` takes its data as ``arg2``:

        >>> (df.pipe(h)
        ...    .pipe(g, arg1=a)
        ...    .pipe((func, 'arg2'), arg1=a, arg3=c)
        ...  )
        """
        return cudf.core.common.pipe(self, func, *args, **kwargs)

    @annotate("SCATTER_BY_MAP", color="green", domain="cudf_python")
    def scatter_by_map(
        self, map_index, map_size=None, keep_index=True, **kwargs
    ):
        """Scatter to a list of dataframes.

        Uses map_index to determine the destination
        of each row of the original DataFrame.

        Parameters
        ----------
        map_index : Series, str or list-like
            Scatter assignment for each row
        map_size : int
            Length of output list. Must be >= uniques in map_index
        keep_index : bool
            Conserve original index values for each row

        Returns
        -------
        A list of cudf.DataFrame objects.
        """

        # map_index might be a column name or array,
        # make it a Column
        if isinstance(map_index, str):
            map_index = self._data[map_index]
        elif isinstance(map_index, cudf.Series):
            map_index = map_index._column
        else:
            map_index = as_column(map_index)

        # Convert float to integer
        if map_index.dtype.kind == "f":
            map_index = map_index.astype(np.int32)

        # Convert string or categorical to integer
        if isinstance(map_index, cudf.core.column.StringColumn):
            map_index = map_index.as_categorical_column(
                "category"
            ).as_numerical
            warnings.warn(
                "Using StringColumn for map_index in scatter_by_map. "
                "Use an integer array/column for better performance."
            )
        elif isinstance(map_index, cudf.core.column.CategoricalColumn):
            map_index = map_index.as_numerical
            warnings.warn(
                "Using CategoricalColumn for map_index in scatter_by_map. "
                "Use an integer array/column for better performance."
            )

        if kwargs.get("debug", False) == 1 and map_size is not None:
            count = map_index.distinct_count()
            if map_size < count:
                raise ValueError(
                    f"ERROR: map_size must be >= {count} (got {map_size})."
                )

        tables = self._partition(map_index, map_size, keep_index)

        return tables

    def dropna(
        self, axis=0, how="any", thresh=None, subset=None, inplace=False
    ):
        """
        Drops rows (or columns) containing nulls from a Column.

        Parameters
        ----------
        axis : {0, 1}, optional
            Whether to drop rows (axis=0, default) or columns (axis=1)
            containing nulls.
        how : {"any", "all"}, optional
            Specifies how to decide whether to drop a row (or column).
            any (default) drops rows (or columns) containing at least
            one null value. all drops only rows (or columns) containing
            *all* null values.
        thresh: int, optional
            If specified, then drops every row (or column) containing
            less than `thresh` non-null values
        subset : list, optional
            List of columns to consider when dropping rows (all columns
            are considered by default). Alternatively, when dropping
            columns, subset is a list of rows to consider.
        inplace : bool, default False
            If True, do operation inplace and return None.

        Returns
        -------
        Copy of the DataFrame with rows/columns containing nulls dropped.

        See also
        --------
        cudf.core.dataframe.DataFrame.isna
            Indicate null values.

        cudf.core.dataframe.DataFrame.notna
            Indicate non-null values.

        cudf.core.dataframe.DataFrame.fillna
            Replace null values.

        cudf.core.series.Series.dropna
            Drop null values.

        cudf.core.index.Index.dropna
            Drop null indices.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"name": ['Alfred', 'Batman', 'Catwoman'],
        ...                    "toy": ['Batmobile', None, 'Bullwhip'],
        ...                    "born": [np.datetime64("1940-04-25"),
        ...                             np.datetime64("NaT"),
        ...                             np.datetime64("NaT")]})
        >>> df
               name        toy                 born
        0    Alfred  Batmobile  1940-04-25 00:00:00
        1    Batman       <NA>                 <NA>
        2  Catwoman   Bullwhip                 <NA>

        Drop the rows where at least one element is null.

        >>> df.dropna()
             name        toy       born
        0  Alfred  Batmobile 1940-04-25

        Drop the columns where at least one element is null.

        >>> df.dropna(axis='columns')
               name
        0    Alfred
        1    Batman
        2  Catwoman

        Drop the rows where all elements are null.

        >>> df.dropna(how='all')
               name        toy                 born
        0    Alfred  Batmobile  1940-04-25 00:00:00
        1    Batman       <NA>                 <NA>
        2  Catwoman   Bullwhip                 <NA>

        Keep only the rows with at least 2 non-null values.

        >>> df.dropna(thresh=2)
               name        toy                 born
        0    Alfred  Batmobile  1940-04-25 00:00:00
        2  Catwoman   Bullwhip                 <NA>

        Define in which columns to look for null values.

        >>> df.dropna(subset=['name', 'born'])
             name        toy       born
        0  Alfred  Batmobile 1940-04-25

        Keep the DataFrame with valid entries in the same variable.

        >>> df.dropna(inplace=True)
        >>> df
             name        toy       born
        0  Alfred  Batmobile 1940-04-25
        """
        if axis == 0:
            result = self._drop_na_rows(
                how=how, subset=subset, thresh=thresh, drop_nan=True
            )
        else:
            result = self._drop_na_columns(
                how=how, subset=subset, thresh=thresh
            )

        return self._mimic_inplace(result, inplace=inplace)

    def fillna(
        self, value=None, method=None, axis=None, inplace=False, limit=None
    ):
        """Fill null values with ``value`` or specified ``method``.

        Parameters
        ----------
        value : scalar, Series-like or dict
            Value to use to fill nulls. If Series-like, null values
            are filled with values in corresponding indices.
            A dict can be used to provide different values to fill nulls
            in different columns. Cannot be used with ``method``.

        method : {'ffill', 'bfill'}, default None
            Method to use for filling null values in the dataframe or series.
            `ffill` propagates the last non-null values forward to the next
            non-null value. `bfill` propagates backward with the next non-null
            value. Cannot be used with ``value``.

        Returns
        -------
        result : DataFrame
            Copy with nulls filled.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, None], 'b': [3, None, 5]})
        >>> df
              a     b
        0     1     3
        1     2  <NA>
        2  <NA>     5
        >>> df.fillna(4)
           a  b
        0  1  3
        1  2  4
        2  4  5
        >>> df.fillna({'a': 3, 'b': 4})
           a  b
        0  1  3
        1  2  4
        2  3  5

        ``fillna`` on a Series object:

        >>> ser = cudf.Series(['a', 'b', None, 'c'])
        >>> ser
        0       a
        1       b
        2    <NA>
        3       c
        dtype: object
        >>> ser.fillna('z')
        0    a
        1    b
        2    z
        3    c
        dtype: object

        ``fillna`` can also supports inplace operation:

        >>> ser.fillna('z', inplace=True)
        >>> ser
        0    a
        1    b
        2    z
        3    c
        dtype: object
        >>> df.fillna({'a': 3, 'b': 4}, inplace=True)
        >>> df
           a  b
        0  1  3
        1  2  4
        2  3  5

        ``fillna`` specified with fill ``method``

        >>> ser = cudf.Series([1, None, None, 2, 3, None, None])
        >>> ser.fillna(method='ffill')
        0    1
        1    1
        2    1
        3    2
        4    3
        5    3
        6    3
        dtype: int64
        >>> ser.fillna(method='bfill')
        0       1
        1       2
        2       2
        3       2
        4       3
        5    <NA>
        6    <NA>
        dtype: int64
        """
        if limit is not None:
            raise NotImplementedError("The limit keyword is not supported")
        if axis:
            raise NotImplementedError("The axis keyword is not supported")

        if value is not None and method is not None:
            raise ValueError("Cannot specify both 'value' and 'method'.")

        if method and method not in {"ffill", "bfill"}:
            raise NotImplementedError(f"Fill method {method} is not supported")

        if isinstance(value, cudf.Series):
            value = value.reindex(self._data.names)
        elif isinstance(value, cudf.DataFrame):
            if not self.index.equals(value.index):
                value = value.reindex(self.index)
            else:
                value = value
        elif not isinstance(value, abc.Mapping):
            value = {name: copy.deepcopy(value) for name in self._data.names}
        elif isinstance(value, abc.Mapping):
            value = {
                key: value.reindex(self.index)
                if isinstance(value, cudf.Series)
                else value
                for key, value in value.items()
            }

        copy_data = self._data.copy(deep=True)

        for name in copy_data.keys():
            should_fill = (
                name in value
                and not libcudf.scalar._is_null_host_scalar(value[name])
            ) or method is not None
            if should_fill:
                copy_data[name] = copy_data[name].fillna(value[name], method)
        result = self._from_table(Frame(copy_data, self._index))

        return self._mimic_inplace(result, inplace=inplace)

    def _drop_na_rows(
        self, how="any", subset=None, thresh=None, drop_nan=False
    ):
        """
        Drops null rows from `self`.

        how : {"any", "all"}, optional
            Specifies how to decide whether to drop a row.
            any (default) drops rows containing at least
            one null value. all drops only rows containing
            *all* null values.
        subset : list, optional
            List of columns to consider when dropping rows.
        thresh: int, optional
            If specified, then drops every row containing
            less than `thresh` non-null values.
        """
        if subset is None:
            subset = self._column_names
        elif (
            not np.iterable(subset)
            or isinstance(subset, str)
            or isinstance(subset, tuple)
            and subset in self._data.names
        ):
            subset = (subset,)
        diff = set(subset) - set(self._data)
        if len(diff) != 0:
            raise KeyError(f"columns {diff} do not exist")
        subset_cols = [
            name for name, col in self._data.items() if name in subset
        ]
        if len(subset_cols) == 0:
            return self.copy(deep=True)

        frame = self.copy(deep=False)
        if drop_nan:
            for name, col in frame._data.items():
                if name in subset and isinstance(
                    col, cudf.core.column.NumericalColumn
                ):
                    frame._data[name] = col.nans_to_nulls()
                else:
                    frame._data[name] = col

        result = frame.__class__._from_table(
            libcudf.stream_compaction.drop_nulls(
                frame, how=how, keys=subset, thresh=thresh
            )
        )
        result._copy_type_metadata(frame)
        return result

    def _drop_na_columns(self, how="any", subset=None, thresh=None):
        """
        Drop columns containing nulls
        """
        out_cols = []

        if subset is None:
            df = self
        else:
            df = self.take(subset)

        if thresh is None:
            if how == "all":
                thresh = 1
            else:
                thresh = len(df)

        for col in self._data.names:
            no_threshold_valid_count = (
                len(df[col]) - df[col].nans_to_nulls().null_count
            ) < thresh
            if no_threshold_valid_count:
                continue
            out_cols.append(col)

        return self[out_cols]

    def _apply_boolean_mask(self, boolean_mask):
        """
        Applies boolean mask to each row of `self`,
        rows corresponding to `False` is dropped
        """
        boolean_mask = as_column(boolean_mask)

        result = self.__class__._from_table(
            libcudf.stream_compaction.apply_boolean_mask(
                self, as_column(boolean_mask)
            )
        )
        result._copy_type_metadata(self)
        return result

    def _quantiles(
        self,
        q,
        interpolation="LINEAR",
        is_sorted=False,
        column_order=(),
        null_precedence=(),
    ):
        interpolation = libcudf.types.Interpolation[interpolation]

        is_sorted = libcudf.types.Sorted["YES" if is_sorted else "NO"]

        column_order = [libcudf.types.Order[key] for key in column_order]

        null_precedence = [
            libcudf.types.NullOrder[key] for key in null_precedence
        ]

        result = self.__class__._from_table(
            libcudf.quantiles.quantiles(
                self,
                q,
                interpolation,
                is_sorted,
                column_order,
                null_precedence,
            )
        )

        result._copy_type_metadata(self)
        return result

    def rank(
        self,
        axis=0,
        method="average",
        numeric_only=None,
        na_option="keep",
        ascending=True,
        pct=False,
    ):
        """
        Compute numerical data ranks (1 through n) along axis.
        By default, equal values are assigned a rank that is the average of the
        ranks of those values.

        Parameters
        ----------
        axis : {0 or 'index'}, default 0
            Index to direct ranking.
        method : {'average', 'min', 'max', 'first', 'dense'}, default 'average'
            How to rank the group of records that have the same value
            (i.e. ties):
            * average: average rank of the group
            * min: lowest rank in the group
            * max: highest rank in the group
            * first: ranks assigned in order they appear in the array
            * dense: like 'min', but rank always increases by 1 between groups.
        numeric_only : bool, optional
            For DataFrame objects, rank only numeric columns if set to True.
        na_option : {'keep', 'top', 'bottom'}, default 'keep'
            How to rank NaN values:
            * keep: assign NaN rank to NaN values
            * top: assign smallest rank to NaN values if ascending
            * bottom: assign highest rank to NaN values if ascending.
        ascending : bool, default True
            Whether or not the elements should be ranked in ascending order.
        pct : bool, default False
            Whether or not to display the returned rankings in percentile
            form.

        Returns
        -------
        same type as caller
            Return a Series or DataFrame with data ranks as values.
        """
        if method not in {"average", "min", "max", "first", "dense"}:
            raise KeyError(method)

        method_enum = libcudf.sort.RankMethod[method.upper()]
        if na_option not in {"keep", "top", "bottom"}:
            raise ValueError(
                "na_option must be one of 'keep', 'top', or 'bottom'"
            )

        if axis not in (0, "index"):
            raise NotImplementedError(
                f"axis must be `0`/`index`, "
                f"axis={axis} is not yet supported in rank"
            )

        source = self
        if numeric_only:
            numeric_cols = (
                name
                for name in self._data.names
                if _is_non_decimal_numeric_dtype(self._data[name])
            )
            source = self._get_columns_by_label(numeric_cols)
            if source.empty:
                return source.astype("float64")

        out_rank_table = libcudf.sort.rank_columns(
            source, method_enum, na_option, ascending, pct
        )

        return self._from_table(out_rank_table).astype(np.float64)

    def repeat(self, repeats, axis=None):
        """Repeats elements consecutively.

        Returns a new object of caller type(DataFrame/Series/Index) where each
        element of the current object is repeated consecutively a given
        number of times.

        Parameters
        ----------
        repeats : int, or array of ints
            The number of repetitions for each element. This should
            be a non-negative integer. Repeating 0 times will return
            an empty object.

        Returns
        -------
        Series/DataFrame/Index
            A newly created object of same type as caller
            with repeated elements.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        >>> df
           a   b
        0  1  10
        1  2  20
        2  3  30
        >>> df.repeat(3)
           a   b
        0  1  10
        0  1  10
        0  1  10
        1  2  20
        1  2  20
        1  2  20
        2  3  30
        2  3  30
        2  3  30

        Repeat on Series

        >>> s = cudf.Series([0, 2])
        >>> s
        0    0
        1    2
        dtype: int64
        >>> s.repeat([3, 4])
        0    0
        0    0
        0    0
        1    2
        1    2
        1    2
        1    2
        dtype: int64
        >>> s.repeat(2)
        0    0
        0    0
        1    2
        1    2
        dtype: int64

        Repeat on Index

        >>> index = cudf.Index([10, 22, 33, 55])
        >>> index
        Int64Index([10, 22, 33, 55], dtype='int64')
        >>> index.repeat(5)
        Int64Index([10, 10, 10, 10, 10, 22, 22, 22, 22, 22, 33,
                    33, 33, 33, 33, 55, 55, 55, 55, 55],
                dtype='int64')
        """
        if axis is not None:
            raise NotImplementedError(
                "Only axis=`None` supported at this time."
            )

        return self._repeat(repeats)

    def _repeat(self, count):
        if not is_scalar(count):
            count = as_column(count)

        result = self.__class__._from_table(
            libcudf.filling.repeat(self, count)
        )

        result._copy_type_metadata(self)
        return result

    def _reverse(self):
        result = self.__class__._from_table(libcudf.copying.reverse(self))
        return result

    def _fill(self, fill_values, begin, end, inplace):
        col_and_fill = zip(self._columns, fill_values)

        if not inplace:
            data_columns = (c._fill(v, begin, end) for (c, v) in col_and_fill)
            data = zip(self._column_names, data_columns)
            return self.__class__._from_table(Frame(data, self._index))

        for (c, v) in col_and_fill:
            c.fill(v, begin, end, inplace=True)

        return self

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        """Shift values by `periods` positions.
        """
        assert axis in (None, 0) and freq is None
        return self._shift(periods)

    def _shift(self, offset, fill_value=None):
        data_columns = (col.shift(offset, fill_value) for col in self._columns)
        data = zip(self._column_names, data_columns)
        return self.__class__._from_table(Frame(data, self._index))

    def __array__(self, dtype=None):
        raise TypeError(
            "Implicit conversion to a host NumPy array via __array__ is not "
            "allowed, To explicitly construct a GPU array, consider using "
            "cupy.asarray(...)\nTo explicitly construct a "
            "host array, consider using .to_array()"
        )

    def __arrow_array__(self, type=None):
        raise TypeError(
            "Implicit conversion to a host PyArrow Array via __arrow_array__ "
            "is not allowed, To explicitly construct a PyArrow Array, "
            "consider using .to_arrow()"
        )

    def round(self, decimals=0, how="half_even"):
        """
        Round a DataFrame to a variable number of decimal places.

        Parameters
        ----------
        decimals : int, dict, Series
            Number of decimal places to round each column to. If an int is
            given, round each column to the same number of places.
            Otherwise dict and Series round to variable numbers of places.
            Column names should be in the keys if `decimals` is a
            dict-like, or in the index if `decimals` is a Series. Any
            columns not included in `decimals` will be left as is. Elements
            of `decimals` which are not columns of the input will be
            ignored.
        how : str, optional
            Type of rounding. Can be either "half_even" (default)
            of "half_up" rounding.

        Returns
        -------
        DataFrame
            A DataFrame with the affected columns rounded to the specified
            number of decimal places.

        Examples
        --------
        >>> df = cudf.DataFrame(
                [(.21, .32), (.01, .67), (.66, .03), (.21, .18)],
        ...     columns=['dogs', 'cats']
        ... )
        >>> df
            dogs  cats
        0  0.21  0.32
        1  0.01  0.67
        2  0.66  0.03
        3  0.21  0.18

        By providing an integer each column is rounded to the same number
        of decimal places

        >>> df.round(1)
            dogs  cats
        0   0.2   0.3
        1   0.0   0.7
        2   0.7   0.0
        3   0.2   0.2

        With a dict, the number of places for specific columns can be
        specified with the column names as key and the number of decimal
        places as value

        >>> df.round({'dogs': 1, 'cats': 0})
            dogs  cats
        0   0.2   0.0
        1   0.0   1.0
        2   0.7   0.0
        3   0.2   0.0

        Using a Series, the number of places for specific columns can be
        specified with the column names as index and the number of
        decimal places as value

        >>> decimals = cudf.Series([0, 1], index=['cats', 'dogs'])
        >>> df.round(decimals)
            dogs  cats
        0   0.2   0.0
        1   0.0   1.0
        2   0.7   0.0
        3   0.2   0.0
        """

        if isinstance(decimals, cudf.Series):
            decimals = decimals.to_pandas()

        if isinstance(decimals, (dict, pd.Series)):
            if (
                isinstance(decimals, pd.Series)
                and not decimals.index.is_unique
            ):
                raise ValueError("Index of decimals must be unique")

            cols = {
                name: col.round(decimals[name], how=how)
                if (
                    name in decimals.keys()
                    and _is_non_decimal_numeric_dtype(col.dtype)
                )
                else col.copy(deep=True)
                for name, col in self._data.items()
            }
        elif isinstance(decimals, int):
            cols = {
                name: col.round(decimals, how=how)
                if _is_non_decimal_numeric_dtype(col.dtype)
                else col.copy(deep=True)
                for name, col in self._data.items()
            }
        else:
            raise TypeError(
                "decimals must be an integer, a dict-like or a Series"
            )

        return self.__class__._from_table(
            Frame(
                data=cudf.core.column_accessor.ColumnAccessor(
                    cols,
                    multiindex=self._data.multiindex,
                    level_names=self._data.level_names,
                )
            ),
            index=self._index,
        )

    @annotate("SAMPLE", color="orange", domain="cudf_python")
    def sample(
        self,
        n=None,
        frac=None,
        replace=False,
        weights=None,
        random_state=None,
        axis=None,
        keep_index=True,
    ):
        """Return a random sample of items from an axis of object.

        You can use random_state for reproducibility.

        Parameters
        ----------
        n : int, optional
            Number of items from axis to return. Cannot be used with frac.
            Default = 1 if frac = None.
        frac : float, optional
            Fraction of axis items to return. Cannot be used with n.
        replace : bool, default False
            Allow or disallow sampling of the same row more than once.
            replace == True is not yet supported for axis = 1/"columns"
        weights : str or ndarray-like, optional
            Only supported for axis=1/"columns"
        random_state : int, numpy RandomState or None, default None
            Seed for the random number generator (if int), or None.
            If None, a random seed will be chosen.
            if RandomState, seed will be extracted from current state.
        axis : {0 or ‘index’, 1 or ‘columns’, None}, default None
            Axis to sample. Accepts axis number or name.
            Default is stat axis for given data type
            (0 for Series and DataFrames). Series and Index doesn't
            support axis=1.

        Returns
        -------
        Series or DataFrame or Index
            A new object of same type as caller containing n items
            randomly sampled from the caller object.

        Examples
        --------
        >>> import cudf as cudf
        >>> df = cudf.DataFrame({"a":{1, 2, 3, 4, 5}})
        >>> df.sample(3)
           a
        1  2
        3  4
        0  1

        >>> sr = cudf.Series([1, 2, 3, 4, 5])
        >>> sr.sample(10, replace=True)
        1    4
        3    1
        2    4
        0    5
        0    1
        4    5
        4    1
        0    2
        0    3
        3    2
        dtype: int64

        >>> df = cudf.DataFrame(
        ... {"a":[1, 2], "b":[2, 3], "c":[3, 4], "d":[4, 5]})
        >>> df.sample(2, axis=1)
           a  c
        0  1  3
        1  2  4
        """

        if frac is not None and frac > 1 and not replace:
            raise ValueError(
                "Replace has to be set to `True` "
                "when upsampling the population `frac` > 1."
            )
        elif frac is not None and n is not None:
            raise ValueError(
                "Please enter a value for `frac` OR `n`, not both"
            )

        if frac is None and n is None:
            n = 1
        elif frac is not None:
            if axis is None or axis == 0 or axis == "index":
                n = int(round(self.shape[0] * frac))
            else:
                n = int(round(self.shape[1] * frac))

        if axis is None or axis == 0 or axis == "index":
            if n > 0 and self.shape[0] == 0:
                raise ValueError(
                    "Cannot take a sample larger than 0 when axis is empty"
                )

            if not replace and n > self.shape[0]:
                raise ValueError(
                    "Cannot take a larger sample than population "
                    "when 'replace=False'"
                )

            if weights is not None:
                raise NotImplementedError(
                    "weights is not yet supported for axis=0/index"
                )

            if random_state is None:
                seed = np.random.randint(
                    np.iinfo(np.int64).max, dtype=np.int64
                )
            elif isinstance(random_state, np.random.mtrand.RandomState):
                _, keys, pos, _, _ = random_state.get_state()
                seed = 0 if pos >= len(keys) else pos
            else:
                seed = np.int64(random_state)

            result = self._from_table(
                libcudf.copying.sample(
                    self,
                    n=n,
                    replace=replace,
                    seed=seed,
                    keep_index=keep_index,
                )
            )
            result._copy_type_metadata(self)

            return result
        else:
            if len(self.shape) != 2:
                raise ValueError(
                    f"No axis named {axis} for "
                    f"object type {self.__class__}"
                )

            if replace:
                raise NotImplementedError(
                    "Sample is not supported for "
                    f"axis {axis} when 'replace=True'"
                )

            if n > 0 and self.shape[1] == 0:
                raise ValueError(
                    "Cannot take a sample larger than 0 when axis is empty"
                )

            columns = np.asarray(self._data.names)
            if not replace and n > columns.size:
                raise ValueError(
                    "Cannot take a larger sample "
                    "than population when 'replace=False'"
                )

            if weights is not None:
                if is_column_like(weights):
                    weights = np.asarray(weights)
                else:
                    raise ValueError(
                        "Strings can only be passed to weights "
                        "when sampling from rows on a DataFrame"
                    )

                if columns.size != len(weights):
                    raise ValueError(
                        "Weights and axis to be sampled must be of same length"
                    )

                total_weight = weights.sum()
                if total_weight != 1:
                    if not isinstance(weights.dtype, float):
                        weights = weights.astype("float64")
                    weights = weights / total_weight

            np.random.seed(random_state)
            gather_map = np.random.choice(
                columns, size=n, replace=replace, p=weights
            )

            if isinstance(self, cudf.MultiIndex):
                # TODO: Need to update this once MultiIndex is refactored,
                # should be able to treat it similar to other Frame object
                result = cudf.Index(self._source_data[gather_map])
            else:
                result = self[gather_map]
                if not keep_index:
                    result.index = None

            return result

    @classmethod
    @annotate("FROM_ARROW", color="orange", domain="cudf_python")
    def from_arrow(cls, data):
        """Convert from PyArrow Table to Frame

        Parameters
        ----------
        data : PyArrow Table

        Raises
        ------
        TypeError for invalid input type.

        Examples
        --------
        >>> import cudf
        >>> import pyarrow as pa
        >>> data = pa.table({"a":[1, 2, 3], "b":[4, 5, 6]})
        >>> cudf.core.frame.Frame.from_arrow(data)
           a  b
        0  1  4
        1  2  5
        2  3  6
        """

        if not isinstance(data, (pa.Table)):
            raise TypeError(
                "To create a multicolumn cudf data, "
                "the data should be an arrow Table"
            )

        column_names = data.column_names
        pandas_dtypes = None
        np_dtypes = None
        if isinstance(data.schema.pandas_metadata, dict):
            metadata = data.schema.pandas_metadata
            pandas_dtypes = {
                col["field_name"]: col["pandas_type"]
                for col in metadata["columns"]
                if "field_name" in col
            }
            np_dtypes = {
                col["field_name"]: col["numpy_type"]
                for col in metadata["columns"]
                if "field_name" in col
            }

        # Currently we don't have support for
        # pyarrow.DictionaryArray -> cudf Categorical column,
        # so handling indices and dictionary as two different columns.
        # This needs be removed once we have hooked libcudf dictionary32
        # with categorical.
        dict_indices = {}
        dict_dictionaries = {}
        dict_ordered = {}
        for field in data.schema:
            if isinstance(field.type, pa.DictionaryType):
                dict_ordered[field.name] = field.type.ordered
                dict_indices[field.name] = pa.chunked_array(
                    [chunk.indices for chunk in data[field.name].chunks],
                    type=field.type.index_type,
                )
                dict_dictionaries[field.name] = pa.chunked_array(
                    [chunk.dictionary for chunk in data[field.name].chunks],
                    type=field.type.value_type,
                )

        # Handle dict arrays
        cudf_category_frame = libcudf.table.Table()
        if len(dict_indices):

            dict_indices_table = pa.table(dict_indices)
            data = data.drop(dict_indices_table.column_names)
            cudf_indices_frame = libcudf.interop.from_arrow(
                dict_indices_table, dict_indices_table.column_names
            )
            # as dictionary size can vary, it can't be a single table
            cudf_dictionaries_columns = {
                name: cudf.core.column.ColumnBase.from_arrow(
                    dict_dictionaries[name]
                )
                for name in dict_dictionaries.keys()
            }

            for name in cudf_indices_frame._data.names:
                codes = cudf_indices_frame._data[name]
                cudf_category_frame._data[name] = build_categorical_column(
                    cudf_dictionaries_columns[name],
                    codes,
                    mask=codes.base_mask,
                    size=codes.size,
                    ordered=dict_ordered[name],
                )

        # Handle non-dict arrays
        cudf_non_category_frame = (
            libcudf.table.Table()
            if data.num_columns == 0
            else libcudf.interop.from_arrow(data, data.column_names)
        )

        if (
            cudf_non_category_frame._num_columns > 0
            and cudf_category_frame._num_columns > 0
        ):
            result = cudf_non_category_frame
            for name in cudf_category_frame._data.names:
                result._data[name] = cudf_category_frame._data[name]
        elif cudf_non_category_frame._num_columns > 0:
            result = cudf_non_category_frame
        else:
            result = cudf_category_frame

        # There are some special cases that need to be handled
        # based on metadata.
        if pandas_dtypes:
            for name in result._data.names:
                dtype = None
                if (
                    len(result._data[name]) == 0
                    and pandas_dtypes[name] == "categorical"
                ):
                    # When pandas_dtype is a categorical column and the size
                    # of column is 0(i.e., empty) then we will have an
                    # int8 column in result._data[name] returned by libcudf,
                    # which needs to be type-casted to 'category' dtype.
                    dtype = "category"
                elif (
                    pandas_dtypes[name] == "empty"
                    and np_dtypes[name] == "object"
                ):
                    # When a string column has all null values, pandas_dtype is
                    # is specified as 'empty' and np_dtypes as 'object',
                    # hence handling this special case to type-cast the empty
                    # float column to str column.
                    dtype = np_dtypes[name]
                elif pandas_dtypes[
                    name
                ] == "object" and cudf.utils.dtypes.is_struct_dtype(
                    np_dtypes[name]
                ):
                    # Incase of struct column, libcudf is not aware of names of
                    # struct fields, hence renaming the struct fields is
                    # necessary by extracting the field names from arrow
                    # struct types.
                    result._data[name] = result._data[name]._rename_fields(
                        [field.name for field in data[name].type]
                    )

                if dtype is not None:
                    result._data[name] = result._data[name].astype(dtype)

        result = libcudf.table.Table(
            result._data.select_by_label(column_names)
        )

        return cls._from_table(result)

    @annotate("TO_ARROW", color="orange", domain="cudf_python")
    def to_arrow(self):
        """
        Convert to arrow Table

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame(
        ...     {"a":[1, 2, 3], "b":[4, 5, 6]}, index=[1, 2, 3])
        >>> df.to_arrow()
        pyarrow.Table
        a: int64
        b: int64
        index: int64
        """
        return pa.Table.from_pydict(
            {name: col.to_arrow() for name, col in self._data.items()}
        )

    def drop_duplicates(
        self,
        subset=None,
        keep="first",
        nulls_are_equal=True,
        ignore_index=False,
    ):
        """
        Drops rows in frame as per duplicate rows in `subset` columns from
        self.

        subset : list, optional
            List of columns to consider when dropping rows.
        keep : ["first", "last", False] first will keep first of duplicate,
            last will keep last of the duplicate and False drop all
            duplicate
        nulls_are_equal: null elements are considered equal to other null
            elements
        ignore_index: bool, default False
            If True, the resulting axis will be labeled 0, 1, …, n - 1.
        """
        if subset is None:
            subset = self._column_names
        elif (
            not np.iterable(subset)
            or isinstance(subset, str)
            or isinstance(subset, tuple)
            and subset in self._data.names
        ):
            subset = (subset,)
        diff = set(subset) - set(self._data)
        if len(diff) != 0:
            raise KeyError(f"columns {diff} do not exist")
        subset_cols = [name for name in self._column_names if name in subset]
        if len(subset_cols) == 0:
            return self.copy(deep=True)

        result = self._from_table(
            libcudf.stream_compaction.drop_duplicates(
                self,
                keys=subset,
                keep=keep,
                nulls_are_equal=nulls_are_equal,
                ignore_index=ignore_index,
            )
        )

        result._copy_type_metadata(self)
        return result

    def replace(self, to_replace: Any, replacement: Any) -> Frame:
        if not (to_replace is None and replacement is None):
            copy_data = self._data.copy(deep=False)
            (
                all_na_per_column,
                to_replace_per_column,
                replacements_per_column,
            ) = _get_replacement_values_for_columns(
                to_replace=to_replace,
                value=replacement,
                columns_dtype_map={
                    col: copy_data._data[col].dtype for col in copy_data._data
                },
            )

            for name, col in copy_data.items():
                try:
                    copy_data[name] = col.find_and_replace(
                        to_replace_per_column[name],
                        replacements_per_column[name],
                        all_na_per_column[name],
                    )
                except (KeyError, OverflowError):
                    # We need to create a deep copy if :
                    # i. `find_and_replace` was not successful or any of
                    #    `to_replace_per_column`, `replacements_per_column`,
                    #    `all_na_per_column` don't contain the `name`
                    #    that exists in `copy_data`.
                    # ii. There is an OverflowError while trying to cast
                    #     `to_replace_per_column` to `replacements_per_column`.
                    copy_data[name] = col.copy(deep=True)
        else:
            copy_data = self._data.copy(deep=True)

        result = self._from_table(Frame(copy_data, self._index))

        return result

    def _copy_type_metadata(
        self, other: Frame, include_index: bool = True
    ) -> Frame:
        """
        Copy type metadata from each column of `other` to the corresponding
        column of `self`.
        See `ColumnBase._with_type_metadata` for more information.
        """
        for name, col, other_col in zip(
            self._data.keys(), self._data.values(), other._data.values()
        ):
            self._data.set_by_label(
                name, col._with_type_metadata(other_col.dtype), validate=False
            )

        if include_index:
            if self._index is not None and other._index is not None:
                self._index._copy_type_metadata(other._index)
                # When other._index is a CategoricalIndex, there is
                if isinstance(
                    other._index, cudf.core.index.CategoricalIndex
                ) and not isinstance(
                    self._index, cudf.core.index.CategoricalIndex
                ):
                    self._index = cudf.core.index.Index._from_table(
                        self._index
                    )

        return self

    def _copy_interval_data(self, other, include_index=True):
        for name, col, other_col in zip(
            self._data.keys(), self._data.values(), other._data.values()
        ):
            if isinstance(other_col, cudf.core.column.IntervalColumn):
                self._data[name] = cudf.core.column.IntervalColumn(col)

    def _postprocess_columns(self, other, include_index=True):
        self._copy_categories(other, include_index=include_index)
        self._copy_struct_names(other, include_index=include_index)
        self._copy_interval_data(other, include_index=include_index)

    def _unaryop(self, op):
        data_columns = (col.unary_operator(op) for col in self._columns)
        data = zip(self._column_names, data_columns)
        return self.__class__._from_table(Frame(data, self._index))

    def isnull(self):
        """
        Identify missing values.

        Return a boolean same-sized object indicating if
        the values are ``<NA>``. ``<NA>`` values gets mapped to
        ``True`` values. Everything else gets mapped to
        ``False`` values. ``<NA>`` values include:

        * Values where null mask is set.
        * ``NaN`` in float dtype.
        * ``NaT`` in datetime64 and timedelta64 types.

        Characters such as empty strings ``''`` or
        ``inf`` incase of float are not
        considered ``<NA>`` values.

        Returns
        -------
        DataFrame/Series/Index
            Mask of bool values for each element in
            the object that indicates whether an element is an NA value.

        Examples
        --------

        Show which entries in a DataFrame are NA.

        >>> import cudf
        >>> import numpy as np
        >>> import pandas as pd
        >>> df = cudf.DataFrame({'age': [5, 6, np.NaN],
        ...                    'born': [pd.NaT, pd.Timestamp('1939-05-27'),
        ...                             pd.Timestamp('1940-04-25')],
        ...                    'name': ['Alfred', 'Batman', ''],
        ...                    'toy': [None, 'Batmobile', 'Joker']})
        >>> df
            age                        born    name        toy
        0     5                        <NA>  Alfred       <NA>
        1     6  1939-05-27 00:00:00.000000  Batman  Batmobile
        2  <NA>  1940-04-25 00:00:00.000000              Joker
        >>> df.isnull()
            age   born   name    toy
        0  False   True  False   True
        1  False  False  False  False
        2   True  False  False  False

        Show which entries in a Series are NA.

        >>> ser = cudf.Series([5, 6, np.NaN, np.inf, -np.inf])
        >>> ser
        0     5.0
        1     6.0
        2    <NA>
        3     Inf
        4    -Inf
        dtype: float64
        >>> ser.isnull()
        0    False
        1    False
        2     True
        3    False
        4    False
        dtype: bool

        Show which entries in an Index are NA.

        >>> idx = cudf.Index([1, 2, None, np.NaN, 0.32, np.inf])
        >>> idx
        Float64Index([1.0, 2.0, <NA>, <NA>, 0.32, Inf], dtype='float64')
        >>> idx.isnull()
        GenericIndex([False, False, True, True, False, False], dtype='bool')
        """
        data_columns = (col.isnull() for col in self._columns)
        data = zip(self._column_names, data_columns)
        return self.__class__._from_table(Frame(data, self._index))

    # Alias for isnull
    isna = isnull

    def notnull(self):
        """
        Identify non-missing values.

        Return a boolean same-sized object indicating if
        the values are not ``<NA>``. Non-missing values get
        mapped to ``True``. ``<NA>`` values get mapped to
        ``False`` values. ``<NA>`` values include:

        * Values where null mask is set.
        * ``NaN`` in float dtype.
        * ``NaT`` in datetime64 and timedelta64 types.

        Characters such as empty strings ``''`` or
        ``inf`` incase of float are not
        considered ``<NA>`` values.

        Returns
        -------
        DataFrame/Series/Index
            Mask of bool values for each element in
            the object that indicates whether an element is not an NA value.

        Examples
        --------

        Show which entries in a DataFrame are NA.

        >>> import cudf
        >>> import numpy as np
        >>> import pandas as pd
        >>> df = cudf.DataFrame({'age': [5, 6, np.NaN],
        ...                    'born': [pd.NaT, pd.Timestamp('1939-05-27'),
        ...                             pd.Timestamp('1940-04-25')],
        ...                    'name': ['Alfred', 'Batman', ''],
        ...                    'toy': [None, 'Batmobile', 'Joker']})
        >>> df
            age                        born    name        toy
        0     5                        <NA>  Alfred       <NA>
        1     6  1939-05-27 00:00:00.000000  Batman  Batmobile
        2  <NA>  1940-04-25 00:00:00.000000              Joker
        >>> df.notnull()
             age   born  name    toy
        0   True  False  True  False
        1   True   True  True   True
        2  False   True  True   True

        Show which entries in a Series are NA.

        >>> ser = cudf.Series([5, 6, np.NaN, np.inf, -np.inf])
        >>> ser
        0     5.0
        1     6.0
        2    <NA>
        3     Inf
        4    -Inf
        dtype: float64
        >>> ser.notnull()
        0     True
        1     True
        2    False
        3     True
        4     True
        dtype: bool

        Show which entries in an Index are NA.

        >>> idx = cudf.Index([1, 2, None, np.NaN, 0.32, np.inf])
        >>> idx
        Float64Index([1.0, 2.0, <NA>, <NA>, 0.32, Inf], dtype='float64')
        >>> idx.notnull()
        GenericIndex([True, True, False, False, True, True], dtype='bool')
        """
        data_columns = (col.notnull() for col in self._columns)
        data = zip(self._column_names, data_columns)
        return self.__class__._from_table(Frame(data, self._index))

    # Alias for notnull
    notna = notnull

    def interleave_columns(self):
        """
        Interleave Series columns of a table into a single column.

        Converts the column major table `cols` into a row major column.

        Parameters
        ----------
        cols : input Table containing columns to interleave.

        Examples
        --------
        >>> df = DataFrame([['A1', 'A2', 'A3'], ['B1', 'B2', 'B3']])
        >>> df
        0    [A1, A2, A3]
        1    [B1, B2, B3]
        >>> df.interleave_columns()
        0    A1
        1    B1
        2    A2
        3    B2
        4    A3
        5    B3

        Returns
        -------
        The interleaved columns as a single column
        """
        if ("category" == self.dtypes).any():
            raise ValueError(
                "interleave_columns does not support 'category' dtype."
            )

        result = self._constructor_sliced(
            libcudf.reshape.interleave_columns(self)
        )

        return result

    def tile(self, count):
        """
        Repeats the rows from `self` DataFrame `count` times to form a
        new DataFrame.

        Parameters
        ----------
        self : input Table containing columns to interleave.
        count : Number of times to tile "rows". Must be non-negative.

        Examples
        --------
        >>> df  = Dataframe([[8, 4, 7], [5, 2, 3]])
        >>> count = 2
        >>> df.tile(df, count)
           0  1  2
        0  8  4  7
        1  5  2  3
        0  8  4  7
        1  5  2  3

        Returns
        -------
        The table containing the tiled "rows".
        """
        result = self.__class__._from_table(libcudf.reshape.tile(self, count))
        result._copy_type_metadata(self)
        return result

    def searchsorted(
        self, values, side="left", ascending=True, na_position="last"
    ):
        """Find indices where elements should be inserted to maintain order

        Parameters
        ----------
        value : Frame (Shape must be consistent with self)
            Values to be hypothetically inserted into Self
        side : str {‘left’, ‘right’} optional, default ‘left‘
            If ‘left’, the index of the first suitable location found is given
            If ‘right’, return the last such index
        ascending : bool optional, default True
            Sorted Frame is in ascending order (otherwise descending)
        na_position : str {‘last’, ‘first’} optional, default ‘last‘
            Position of null values in sorted order

        Returns
        -------
        1-D cupy array of insertion points

        Examples
        --------
        >>> s = cudf.Series([1, 2, 3])
        >>> s.searchsorted(4)
        3
        >>> s.searchsorted([0, 4])
        array([0, 3], dtype=int32)
        >>> s.searchsorted([1, 3], side='left')
        array([0, 2], dtype=int32)
        >>> s.searchsorted([1, 3], side='right')
        array([1, 3], dtype=int32)

        If the values are not monotonically sorted, wrong
        locations may be returned:

        >>> s = cudf.Series([2, 1, 3])
        >>> s.searchsorted(1)
        0   # wrong result, correct would be 1

        >>> df = cudf.DataFrame({'a': [1, 3, 5, 7], 'b': [10, 12, 14, 16]})
        >>> df
           a   b
        0  1  10
        1  3  12
        2  5  14
        3  7  16
        >>> values_df = cudf.DataFrame({'a': [0, 2, 5, 6],
        ... 'b': [10, 11, 13, 15]})
        >>> values_df
           a   b
        0  0  10
        1  2  17
        2  5  13
        3  6  15
        >>> df.searchsorted(values_df, ascending=False)
        array([4, 4, 4, 0], dtype=int32)
        """
        # Call libcudf++ search_sorted primitive

        scalar_flag = None
        if is_scalar(values):
            scalar_flag = True

        if not isinstance(values, Frame):
            values = as_column(values)
            if values.dtype != self.dtype:
                self = self.astype(values.dtype)
            values = values.as_frame()
        outcol = libcudf.search.search_sorted(
            self, values, side, ascending=ascending, na_position=na_position
        )

        # Retrun result as cupy array if the values is non-scalar
        # If values is scalar, result is expected to be scalar.
        result = cupy.asarray(outcol.data_array_view)
        if scalar_flag:
            return result[0].item()
        else:
            return result

    def _get_sorted_inds(self, by=None, ascending=True, na_position="last"):
        """
        Sort by the values.

        Parameters
        ----------
        by: list, optional
            Labels specifying columns to sort by. By default,
            sort by all columns of `self`
        ascending : bool or list of bool, default True
            If True, sort values in ascending order, otherwise descending.
        na_position : {‘first’ or ‘last’}, default ‘last’
            Argument ‘first’ puts NaNs at the beginning, ‘last’ puts NaNs
            at the end.
        Returns
        -------
        out_column_inds : cuDF Column of indices sorted based on input

        Difference from pandas:
        * Support axis='index' only.
        * Not supporting: inplace, kind
        * Ascending can be a list of bools to control per column
        """

        # This needs to be updated to handle list of bools for ascending
        if ascending is True:
            if na_position == "last":
                na_position = 0
            elif na_position == "first":
                na_position = 1
        elif ascending is False:
            if na_position == "last":
                na_position = 1
            elif na_position == "first":
                na_position = 0
        else:
            warnings.warn(
                "When using a sequence of booleans for `ascending`, "
                "`na_position` flag is not yet supported and defaults to "
                "treating nulls as greater than all numbers"
            )
            na_position = 0

        to_sort = (
            self
            if by is None
            else self._get_columns_by_label(by, downcast=False)
        )

        # If given a scalar need to construct a sequence of length # of columns
        if np.isscalar(ascending):
            ascending = [ascending] * to_sort._num_columns

        return libcudf.sort.order_by(to_sort, ascending, na_position)

    def sin(self):
        """
        Get Trigonometric sine, element-wise.

        Returns
        -------
        DataFrame/Series/Index
            Result of the trigonometric operation.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([0.0, 0.32434, 0.5, 45, 90, 180, 360])
        >>> ser
        0      0.00000
        1      0.32434
        2      0.50000
        3     45.00000
        4     90.00000
        5    180.00000
        6    360.00000
        dtype: float64
        >>> ser.sin()
        0    0.000000
        1    0.318683
        2    0.479426
        3    0.850904
        4    0.893997
        5   -0.801153
        6    0.958916
        dtype: float64

        `sin` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [0.0, 5, 10, 15],
        ...                      'second': [100.0, 360, 720, 300]})
        >>> df
           first  second
        0    0.0   100.0
        1    5.0   360.0
        2   10.0   720.0
        3   15.0   300.0
        >>> df.sin()
              first    second
        0  0.000000 -0.506366
        1 -0.958924  0.958916
        2 -0.544021 -0.544072
        3  0.650288 -0.999756

        `sin` operation on Index:

        >>> index = cudf.Index([-0.4, 100, -180, 90])
        >>> index
        Float64Index([-0.4, 100.0, -180.0, 90.0], dtype='float64')
        >>> index.sin()
        Float64Index([-0.3894183423086505, -0.5063656411097588,
                    0.8011526357338306, 0.8939966636005579],
                    dtype='float64')
        """
        return self._unaryop("sin")

    def cos(self):
        """
        Get Trigonometric cosine, element-wise.

        Returns
        -------
        DataFrame/Series/Index
            Result of the trigonometric operation.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([0.0, 0.32434, 0.5, 45, 90, 180, 360])
        >>> ser
        0      0.00000
        1      0.32434
        2      0.50000
        3     45.00000
        4     90.00000
        5    180.00000
        6    360.00000
        dtype: float64
        >>> ser.cos()
        0    1.000000
        1    0.947861
        2    0.877583
        3    0.525322
        4   -0.448074
        5   -0.598460
        6   -0.283691
        dtype: float64

        `cos` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [0.0, 5, 10, 15],
        ...                      'second': [100.0, 360, 720, 300]})
        >>> df
           first  second
        0    0.0   100.0
        1    5.0   360.0
        2   10.0   720.0
        3   15.0   300.0
        >>> df.cos()
              first    second
        0  1.000000  0.862319
        1  0.283662 -0.283691
        2 -0.839072 -0.839039
        3 -0.759688 -0.022097

        `cos` operation on Index:

        >>> index = cudf.Index([-0.4, 100, -180, 90])
        >>> index
        Float64Index([-0.4, 100.0, -180.0, 90.0], dtype='float64')
        >>> index.cos()
        Float64Index([ 0.9210609940028851,  0.8623188722876839,
                    -0.5984600690578581, -0.4480736161291701],
                    dtype='float64')
        """
        return self._unaryop("cos")

    def tan(self):
        """
        Get Trigonometric tangent, element-wise.

        Returns
        -------
        DataFrame/Series/Index
            Result of the trigonometric operation.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([0.0, 0.32434, 0.5, 45, 90, 180, 360])
        >>> ser
        0      0.00000
        1      0.32434
        2      0.50000
        3     45.00000
        4     90.00000
        5    180.00000
        6    360.00000
        dtype: float64
        >>> ser.tan()
        0    0.000000
        1    0.336213
        2    0.546302
        3    1.619775
        4   -1.995200
        5    1.338690
        6   -3.380140
        dtype: float64

        `tan` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [0.0, 5, 10, 15],
        ...                      'second': [100.0, 360, 720, 300]})
        >>> df
           first  second
        0    0.0   100.0
        1    5.0   360.0
        2   10.0   720.0
        3   15.0   300.0
        >>> df.tan()
              first     second
        0  0.000000  -0.587214
        1 -3.380515  -3.380140
        2  0.648361   0.648446
        3 -0.855993  45.244742

        `tan` operation on Index:

        >>> index = cudf.Index([-0.4, 100, -180, 90])
        >>> index
        Float64Index([-0.4, 100.0, -180.0, 90.0], dtype='float64')
        >>> index.tan()
        Float64Index([-0.4227932187381618,  -0.587213915156929,
                    -1.3386902103511544, -1.995200412208242],
                    dtype='float64')
        """
        return self._unaryop("tan")

    def asin(self):
        """
        Get Trigonometric inverse sine, element-wise.

        The inverse of sine so that, if y = x.sin(), then x = y.asin()

        Returns
        -------
        DataFrame/Series/Index
            Result of the trigonometric operation.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([-1, 0, 1, 0.32434, 0.5])
        >>> ser.asin()
        0   -1.570796
        1    0.000000
        2    1.570796
        3    0.330314
        4    0.523599
        dtype: float64

        `asin` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [-1, 0, 0.5],
        ...                      'second': [0.234, 0.3, 0.1]})
        >>> df
           first  second
        0   -1.0   0.234
        1    0.0   0.300
        2    0.5   0.100
        >>> df.asin()
              first    second
        0 -1.570796  0.236190
        1  0.000000  0.304693
        2  0.523599  0.100167

        `asin` operation on Index:

        >>> index = cudf.Index([-1, 0.4, 1, 0.3])
        >>> index
        Float64Index([-1.0, 0.4, 1.0, 0.3], dtype='float64')
        >>> index.asin()
        Float64Index([-1.5707963267948966, 0.41151684606748806,
                    1.5707963267948966, 0.3046926540153975],
                    dtype='float64')
        """
        return self._unaryop("asin")

    def acos(self):
        """
        Get Trigonometric inverse cosine, element-wise.

        The inverse of cos so that, if y = x.cos(), then x = y.acos()

        Returns
        -------
        DataFrame/Series/Index
            Result of the trigonometric operation.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([-1, 0, 1, 0.32434, 0.5])
        >>> ser.acos()
        0    3.141593
        1    1.570796
        2    0.000000
        3    1.240482
        4    1.047198
        dtype: float64

        `acos` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [-1, 0, 0.5],
        ...                      'second': [0.234, 0.3, 0.1]})
        >>> df
           first  second
        0   -1.0   0.234
        1    0.0   0.300
        2    0.5   0.100
        >>> df.acos()
              first    second
        0  3.141593  1.334606
        1  1.570796  1.266104
        2  1.047198  1.470629

        `acos` operation on Index:

        >>> index = cudf.Index([-1, 0.4, 1, 0, 0.3])
        >>> index
        Float64Index([-1.0, 0.4, 1.0, 0.0, 0.3], dtype='float64')
        >>> index.acos()
        Float64Index([ 3.141592653589793, 1.1592794807274085, 0.0,
                    1.5707963267948966,  1.266103672779499],
                    dtype='float64')
        """
        result = self.copy(deep=False)
        for col in result._data:
            min_float_dtype = cudf.utils.dtypes.get_min_float_dtype(
                result._data[col]
            )
            result._data[col] = result._data[col].astype(min_float_dtype)
        result = result._unaryop("acos")
        result = result.mask((result < 0) | (result > np.pi + 1))
        return result

    def atan(self):
        """
        Get Trigonometric inverse tangent, element-wise.

        The inverse of tan so that, if y = x.tan(), then x = y.atan()

        Returns
        -------
        DataFrame/Series/Index
            Result of the trigonometric operation.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([-1, 0, 1, 0.32434, 0.5, -10])
        >>> ser
        0    -1.00000
        1     0.00000
        2     1.00000
        3     0.32434
        4     0.50000
        5   -10.00000
        dtype: float64
        >>> ser.atan()
        0   -0.785398
        1    0.000000
        2    0.785398
        3    0.313635
        4    0.463648
        5   -1.471128
        dtype: float64

        `atan` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [-1, -10, 0.5],
        ...                      'second': [0.234, 0.3, 10]})
        >>> df
           first  second
        0   -1.0   0.234
        1  -10.0   0.300
        2    0.5  10.000
        >>> df.atan()
              first    second
        0 -0.785398  0.229864
        1 -1.471128  0.291457
        2  0.463648  1.471128

        `atan` operation on Index:

        >>> index = cudf.Index([-1, 0.4, 1, 0, 0.3])
        >>> index
        Float64Index([-1.0, 0.4, 1.0, 0.0, 0.3], dtype='float64')
        >>> index.atan()
        Float64Index([-0.7853981633974483,  0.3805063771123649,
                                    0.7853981633974483, 0.0,
                                    0.2914567944778671],
                    dtype='float64')
        """
        return self._unaryop("atan")

    def exp(self):
        """
        Get the exponential of all elements, element-wise.

        Exponential is the inverse of the log function,
        so that x.exp().log() = x

        Returns
        -------
        DataFrame/Series/Index
            Result of the element-wise exponential.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([-1, 0, 1, 0.32434, 0.5, -10, 100])
        >>> ser
        0     -1.00000
        1      0.00000
        2      1.00000
        3      0.32434
        4      0.50000
        5    -10.00000
        6    100.00000
        dtype: float64
        >>> ser.exp()
        0    3.678794e-01
        1    1.000000e+00
        2    2.718282e+00
        3    1.383117e+00
        4    1.648721e+00
        5    4.539993e-05
        6    2.688117e+43
        dtype: float64

        `exp` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [-1, -10, 0.5],
        ...                      'second': [0.234, 0.3, 10]})
        >>> df
           first  second
        0   -1.0   0.234
        1  -10.0   0.300
        2    0.5  10.000
        >>> df.exp()
              first        second
        0  0.367879      1.263644
        1  0.000045      1.349859
        2  1.648721  22026.465795

        `exp` operation on Index:

        >>> index = cudf.Index([-1, 0.4, 1, 0, 0.3])
        >>> index
        Float64Index([-1.0, 0.4, 1.0, 0.0, 0.3], dtype='float64')
        >>> index.exp()
        Float64Index([0.36787944117144233,  1.4918246976412703,
                      2.718281828459045, 1.0,  1.3498588075760032],
                    dtype='float64')
        """
        return self._unaryop("exp")

    def log(self):
        """
        Get the natural logarithm of all elements, element-wise.

        Natural logarithm is the inverse of the exp function,
        so that x.log().exp() = x

        Returns
        -------
        DataFrame/Series/Index
            Result of the element-wise natural logarithm.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([-1, 0, 1, 0.32434, 0.5, -10, 100])
        >>> ser
        0     -1.00000
        1      0.00000
        2      1.00000
        3      0.32434
        4      0.50000
        5    -10.00000
        6    100.00000
        dtype: float64
        >>> ser.log()
        0         NaN
        1        -inf
        2    0.000000
        3   -1.125963
        4   -0.693147
        5         NaN
        6    4.605170
        dtype: float64

        `log` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [-1, -10, 0.5],
        ...                      'second': [0.234, 0.3, 10]})
        >>> df
           first  second
        0   -1.0   0.234
        1  -10.0   0.300
        2    0.5  10.000
        >>> df.log()
              first    second
        0       NaN -1.452434
        1       NaN -1.203973
        2 -0.693147  2.302585

        `log` operation on Index:

        >>> index = cudf.Index([10, 11, 500.0])
        >>> index
        Float64Index([10.0, 11.0, 500.0], dtype='float64')
        >>> index.log()
        Float64Index([2.302585092994046, 2.3978952727983707,
                    6.214608098422191], dtype='float64')
        """
        return self._unaryop("log")

    def sqrt(self):
        """
        Get the non-negative square-root of all elements, element-wise.

        Returns
        -------
        DataFrame/Series/Index
            Result of the non-negative
            square-root of each element.

        Examples
        --------
        >>> import cudf
        >>> import cudf
        >>> ser = cudf.Series([10, 25, 81, 1.0, 100])
        >>> ser
        0     10.0
        1     25.0
        2     81.0
        3      1.0
        4    100.0
        dtype: float64
        >>> ser.sqrt()
        0     3.162278
        1     5.000000
        2     9.000000
        3     1.000000
        4    10.000000
        dtype: float64

        `sqrt` operation on DataFrame:

        >>> df = cudf.DataFrame({'first': [-10.0, 100, 625],
        ...                      'second': [1, 2, 0.4]})
        >>> df
           first  second
        0  -10.0     1.0
        1  100.0     2.0
        2  625.0     0.4
        >>> df.sqrt()
           first    second
        0    NaN  1.000000
        1   10.0  1.414214
        2   25.0  0.632456

        `sqrt` operation on Index:

        >>> index = cudf.Index([-10.0, 100, 625])
        >>> index
        Float64Index([-10.0, 100.0, 625.0], dtype='float64')
        >>> index.sqrt()
        Float64Index([nan, 10.0, 25.0], dtype='float64')
        """
        return self._unaryop("sqrt")

    def _merge(
        self,
        right,
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        how="inner",
        sort=False,
        method="hash",
        indicator=False,
        suffixes=("_x", "_y"),
    ):
        lhs, rhs = self, right
        if how == "right":
            # Merge doesn't support right, so just swap
            how = "left"
            lhs, rhs = right, self
            left_on, right_on = right_on, left_on
            left_index, right_index = right_index, left_index
            suffixes = (suffixes[1], suffixes[0])

        return merge(
            lhs,
            rhs,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            sort=sort,
            method=method,
            indicator=indicator,
            suffixes=suffixes,
        )

    def _is_sorted(self, ascending=None, null_position=None):
        """
        Returns a boolean indicating whether the data of the Frame are sorted
        based on the parameters given. Does not account for the index.

        Parameters
        ----------
        self : Frame
            Frame whose columns are to be checked for sort order
        ascending : None or list-like of booleans
            None or list-like of boolean values indicating expected sort order
            of each column. If list-like, size of list-like must be
            len(columns). If None, all columns expected sort order is set to
            ascending. False (0) - ascending, True (1) - descending.
        null_position : None or list-like of booleans
            None or list-like of boolean values indicating desired order of
            nulls compared to other elements. If list-like, size of list-like
            must be len(columns). If None, null order is set to before. False
            (0) - before, True (1) - after.

        Returns
        -------
        returns : boolean
            Returns True, if sorted as expected by ``ascending`` and
            ``null_position``, False otherwise.
        """
        return libcudf.sort.is_sorted(
            self, ascending=ascending, null_position=null_position
        )

    def _split(self, splits, keep_index=True):
        result = libcudf.copying.table_split(
            self, splits, keep_index=keep_index
        )
        result = [self.__class__._from_table(tbl) for tbl in result]
        return result

    def _encode(self):
        keys, indices = libcudf.transform.table_encode(self)
        keys = self.__class__._from_table(keys)
        return keys, indices

    def _reindex(
        self, columns, dtypes=None, deep=False, index=None, inplace=False
    ):
        """
        Helper for `.reindex`

        Parameters
        ----------
        columns : array-like
            The list of columns to select from the Frame,
            if ``columns`` is a superset of ``Frame.columns`` new
            columns are created.
        dtypes : dict
            Mapping of dtypes for the empty columns being created.
        deep : boolean, optional, default False
            Whether to make deep copy or shallow copy of the columns.
        index : Index or array-like, default None
            The ``index`` to be used to reindex the Frame with.
        inplace : bool, default False
            Whether to perform the operation in place on the data.

        Returns
        -------
        DataFrame
        """
        if dtypes is None:
            dtypes = {}

        df = self
        if index is not None:
            index = cudf.core.index.as_index(index)

            if isinstance(index, cudf.core.MultiIndex):
                idx_dtype_match = (
                    df.index._source_data.dtypes == index._source_data.dtypes
                ).all()
            else:
                idx_dtype_match = df.index.dtype == index.dtype

            if not idx_dtype_match:
                columns = columns if columns is not None else list(df.columns)
                df = cudf.DataFrame()
            else:
                df = cudf.DataFrame(None, index).join(
                    df, how="left", sort=True
                )
                # double-argsort to map back from sorted to unsorted positions
                df = df.take(index.argsort(ascending=True).argsort())

        index = index if index is not None else df.index
        names = columns if columns is not None else list(df.columns)
        cols = {
            name: (
                df._data[name].copy(deep=deep)
                if name in df._data
                else column_empty(
                    dtype=dtypes.get(name, np.float64),
                    masked=True,
                    row_count=len(index),
                )
            )
            for name in names
        }

        result = self.__class__._from_table(
            Frame(
                data=cudf.core.column_accessor.ColumnAccessor(
                    cols,
                    multiindex=self._data.multiindex,
                    level_names=self._data.level_names,
                )
            ),
            index=index,
        )

        return self._mimic_inplace(result, inplace=inplace)


_truediv_int_dtype_corrections = {
    np.int8: np.float32,
    np.int16: np.float32,
    np.int32: np.float32,
    np.int64: np.float64,
    np.uint8: np.float32,
    np.uint16: np.float32,
    np.uint32: np.float64,
    np.uint64: np.float64,
    np.bool_: np.float32,
}


class SingleColumnFrame(Frame):
    """A one-dimensional frame.

    Frames with only a single column share certain logic that is encoded in
    this class.
    """

    @property
    def name(self):
        """The name of this object."""
        return next(iter(self._data.names))

    @name.setter
    def name(self, value):
        self._data[value] = self._data.pop(self.name)

    @property
    def ndim(self):
        """Dimension of the data (always 1)."""
        return 1

    @property
    def shape(self):
        """Returns a tuple representing the dimensionality of the Index.
        """
        return (len(self),)

    def __iter__(self):
        cudf.utils.utils.raise_iteration_error(obj=self)

    def __len__(self):
        return len(self._column)

    def __bool__(self):
        raise TypeError(
            f"The truth value of a {type(self)} is ambiguous. Use "
            "a.empty, a.bool(), a.item(), a.any() or a.all()."
        )

    @property
    def _num_columns(self):
        return 1

    @property
    def _column(self):
        return self._data[self.name]

    @_column.setter
    def _column(self, value):
        self._data[self.name] = value

    @property
    def values(self):
        """
        Return a CuPy representation of the data.

        Returns
        -------
        out : cupy.ndarray
            A device representation of the underlying data.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, -10, 100, 20])
        >>> ser.values
        array([  1, -10, 100,  20])
        >>> type(ser.values)
        <class 'cupy.core.core.ndarray'>
        >>> index = cudf.Index([1, -10, 100, 20])
        >>> index.values
        array([  1, -10, 100,  20])
        >>> type(index.values)
        <class 'cupy.core.core.ndarray'>
        """
        return self._column.values

    @property
    def values_host(self):
        """
        Return a NumPy representation of the data.

        Returns
        -------
        out : numpy.ndarray
            A host representation of the underlying data.

        Examples
        --------
        >>> import cudf
        >>> ser = cudf.Series([1, -10, 100, 20])
        >>> ser.values_host
        array([  1, -10, 100,  20])
        >>> type(ser.values_host)
        <class 'numpy.ndarray'>
        >>> index = cudf.Index([1, -10, 100, 20])
        >>> index.values_host
        array([  1, -10, 100,  20])
        >>> type(index.values_host)
        <class 'numpy.ndarray'>
        """
        return self._column.values_host

    def tolist(self):

        raise TypeError(
            "cuDF does not support conversion to host memory "
            "via the `tolist()` method. Consider using "
            "`.to_arrow().to_pylist()` to construct a Python list."
        )

    to_list = tolist

    def to_gpu_array(self, fillna=None):
        """Get a dense numba device array for the data.

        Parameters
        ----------
        fillna : str or None
            See *fillna* in ``.to_array``.

        Notes
        -----

        if ``fillna`` is ``None``, null values are skipped.  Therefore, the
        output size could be smaller.

        Returns
        -------
        numba.DeviceNDArray

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series([10, 20, 30, 40, 50])
        >>> s
        0    10
        1    20
        2    30
        3    40
        4    50
        dtype: int64
        >>> s.to_gpu_array()
        <numba.cuda.cudadrv.devicearray.DeviceNDArray object at 0x7f1840858890>
        """
        return self._column.to_gpu_array(fillna=fillna)

    @classmethod
    def from_arrow(cls, array):
        """Create from PyArrow Array/ChunkedArray.

        Parameters
        ----------
        array : PyArrow Array/ChunkedArray
            PyArrow Object which has to be converted.

        Raises
        ------
        TypeError for invalid input type.

        Returns
        -------
        SingleColumnFrame

        Examples
        --------
        >>> import cudf
        >>> import pyarrow as pa
        >>> cudf.Index.from_arrow(pa.array(["a", "b", None]))
        StringIndex(['a' 'b' None], dtype='object')
        >>> cudf.Series.from_arrow(pa.array(["a", "b", None]))
        0       a
        1       b
        2    <NA>
        dtype: object
        """
        return cls(cudf.core.column.column.ColumnBase.from_arrow(array))

    def to_arrow(self):
        """
        Convert to a PyArrow Array.

        Returns
        -------
        PyArrow Array

        Examples
        --------
        >>> import cudf
        >>> sr = cudf.Series(["a", "b", None])
        >>> sr.to_arrow()
        <pyarrow.lib.StringArray object at 0x7f796b0e7600>
        [
          "a",
          "b",
          null
        ]
        >>> ind = cudf.Index(["a", "b", None])
        >>> ind.to_arrow()
        <pyarrow.lib.StringArray object at 0x7f796b0e7750>
        [
          "a",
          "b",
          null
        ]
        """
        return self._column.to_arrow()

    @property
    def is_unique(self):
        """Return boolean if values in the object are unique.

        Returns
        -------
        bool
        """
        return self._column.is_unique

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
        """Return boolean if values in the object are monotonic_increasing.

        Returns
        -------
        bool
        """
        return self._column.is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self):
        """Return boolean if values in the object are monotonic_decreasing.

        Returns
        -------
        bool
        """
        return self._column.is_monotonic_decreasing

    @property
    def __cuda_array_interface__(self):
        return self._column.__cuda_array_interface__

    def factorize(self, na_sentinel=-1):
        """Encode the input values as integer labels

        Parameters
        ----------
        na_sentinel : number
            Value to indicate missing category.

        Returns
        --------
        (labels, cats) : (cupy.ndarray, cupy.ndarray or Index)
            - *labels* contains the encoded values
            - *cats* contains the categories in order that the N-th
              item corresponds to the (N-1) code.

        Examples
        --------
        >>> import cudf
        >>> s = cudf.Series(['a', 'a', 'c'])
        >>> codes, uniques = s.factorize()
        >>> codes
        array([0, 0, 1], dtype=int8)
        >>> uniques
        StringIndex(['a' 'c'], dtype='object')
        """
        return cudf.core.algorithms.factorize(self, na_sentinel=na_sentinel)

    @property
    def _copy_construct_defaults(self):
        """A default dictionary of kwargs to be used for copy construction."""
        raise NotImplementedError

    def _copy_construct(self, **kwargs):
        """Shallow copy this object by replacing certain ctor args.
        """
        return self.__class__(**{**self._copy_construct_defaults, **kwargs})

    def _binaryop(
        self,
        other,
        fn,
        fill_value=None,
        reflect=False,
        lhs=None,
        *args,
        **kwargs,
    ):
        """Perform a binary operation between two single column frames.

        Parameters
        ----------
        other : SingleColumnFrame
            The second operand.
        fn : str
            The operation
        fill_value : Any, default None
            The value to replace null values with. If ``None``, nulls are not
            filled before the operation.
        reflect : bool, default False
            If ``True`` the operation is reflected (i.e whether to swap the
            left and right operands).
        lhs : SingleColumnFrame, default None
            The left hand operand. If ``None``, self is used. This parameter
            allows child classes to preprocess the inputs if necessary.

        Returns
        -------
        SingleColumnFrame
            A new instance containing the result of the operation.
        """
        if lhs is None:
            lhs = self

        rhs = self._normalize_binop_value(other)

        if fn == "truediv":
            truediv_type = _truediv_int_dtype_corrections.get(lhs.dtype.type)
            if truediv_type is not None:
                lhs = lhs.astype(truediv_type)

        output_mask = None
        if fill_value is not None:
            if is_scalar(rhs):
                if lhs.nullable:
                    lhs = lhs.fillna(fill_value)
            else:
                # If both columns are nullable, pandas semantics dictate that
                # nulls that are present in both lhs and rhs are not filled.
                if lhs.nullable and rhs.nullable:
                    # Note: lhs is a Frame, while rhs is already a column.
                    lmask = as_column(lhs._column.nullmask)
                    rmask = as_column(rhs.nullmask)
                    output_mask = (lmask | rmask).data
                    lhs = lhs.fillna(fill_value)
                    rhs = rhs.fillna(fill_value)
                elif lhs.nullable:
                    lhs = lhs.fillna(fill_value)
                elif rhs.nullable:
                    rhs = rhs.fillna(fill_value)

        outcol = lhs._column.binary_operator(fn, rhs, reflect=reflect)

        # Get the appropriate name for output operations involving two objects
        # that are Series-like objects. The output shares the lhs's name unless
        # the rhs is a _differently_ named Series-like object.
        if (
            isinstance(other, (SingleColumnFrame, pd.Series, pd.Index))
            and self.name != other.name
        ):
            result_name = None
        else:
            result_name = self.name

        output = lhs._copy_construct(data=outcol, name=result_name)

        if output_mask is not None:
            output._column = output._column.set_mask(output_mask)
        return output

    def _normalize_binop_value(self, other):
        """Returns a *column* (not a Series) or scalar for performing
        binary operations with self._column.
        """
        if isinstance(other, ColumnBase):
            return other
        if isinstance(other, SingleColumnFrame):
            return other._column
        if other is cudf.NA:
            return cudf.Scalar(other, dtype=self.dtype)
        else:
            return self._column.normalize_binop_value(other)

    def _bitwise_binop(self, other, op):
        """Type-coercing wrapper around _binaryop for bitwise operations."""
        # This will catch attempts at bitwise ops on extension dtypes.
        try:
            self_is_bool = np.issubdtype(self.dtype, np.bool_)
            other_is_bool = np.issubdtype(other.dtype, np.bool_)
        except TypeError:
            raise TypeError(
                f"Operation 'bitwise {op}' not supported between "
                f"{self.dtype.type.__name__} and {other.dtype.type.__name__}"
            )

        if (self_is_bool or np.issubdtype(self.dtype, np.integer)) and (
            other_is_bool or np.issubdtype(other.dtype, np.integer)
        ):
            # TODO: This doesn't work on Series (op) DataFrame
            # because dataframe doesn't have dtype
            ser = self._binaryop(other, op)
            if self_is_bool or other_is_bool:
                ser = ser.astype(np.bool_)
            return ser
        else:
            raise TypeError(
                f"Operation 'bitwise {op}' not supported between "
                f"{self.dtype.type.__name__} and {other.dtype.type.__name__}"
            )

    # Binary arithmetic operations.
    def __add__(self, other):
        return self._binaryop(other, "add")

    def __radd__(self, other):
        return self._binaryop(other, "add", reflect=True)

    def __sub__(self, other):
        return self._binaryop(other, "sub")

    def __rsub__(self, other):
        return self._binaryop(other, "sub", reflect=True)

    def __mul__(self, other):
        return self._binaryop(other, "mul")

    def __rmul__(self, other):
        return self._binaryop(other, "mul", reflect=True)

    def __mod__(self, other):
        return self._binaryop(other, "mod")

    def __rmod__(self, other):
        return self._binaryop(other, "mod", reflect=True)

    def __pow__(self, other):
        return self._binaryop(other, "pow")

    def __rpow__(self, other):
        return self._binaryop(other, "pow", reflect=True)

    def __floordiv__(self, other):
        return self._binaryop(other, "floordiv")

    def __rfloordiv__(self, other):
        return self._binaryop(other, "floordiv", reflect=True)

    def __truediv__(self, other):
        if is_decimal_dtype(self.dtype):
            return self._binaryop(other, "div")
        else:
            return self._binaryop(other, "truediv")

    def __rtruediv__(self, other):
        if is_decimal_dtype(self.dtype):
            return self._binaryop(other, "div", reflect=True)
        else:
            return self._binaryop(other, "truediv", reflect=True)

    __div__ = __truediv__

    def __and__(self, other):
        return self._bitwise_binop(other, "and")

    def __or__(self, other):
        return self._bitwise_binop(other, "or")

    def __xor__(self, other):
        return self._bitwise_binop(other, "xor")

    # Binary rich comparison operations.
    def __eq__(self, other):
        return self._binaryop(other, "eq")

    def __ne__(self, other):
        return self._binaryop(other, "ne")

    def __lt__(self, other):
        return self._binaryop(other, "lt")

    def __le__(self, other):
        return self._binaryop(other, "le")

    def __gt__(self, other):
        return self._binaryop(other, "gt")

    def __ge__(self, other):
        return self._binaryop(other, "ge")

    # Unary logical operators
    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self.copy(deep=True)

    def __abs__(self):
        return self._unaryop("abs")


def _get_replacement_values_for_columns(
    to_replace: Any, value: Any, columns_dtype_map: Dict[Any, Any]
) -> Tuple[Dict[Any, bool], Dict[Any, Any], Dict[Any, Any]]:
    """
    Returns a per column mapping for the values to be replaced, new
    values to be replaced with and if all the values are empty.

    Parameters
    ----------
    to_replace : numeric, str, list-like or dict
        Contains the values to be replaced.
    value : numeric, str, list-like, or dict
        Contains the values to replace `to_replace` with.
    columns_dtype_map : dict
        A column to dtype mapping representing dtype of columns.

    Returns
    -------
    all_na_columns : dict
        A dict mapping of all columns if they contain all na values
    to_replace_columns : dict
        A dict mapping of all columns and the existing values that
        have to be replaced.
    values_columns : dict
        A dict mapping of all columns and the corresponding values
        to be replaced with.
    """
    to_replace_columns: Dict[Any, Any] = {}
    values_columns: Dict[Any, Any] = {}
    all_na_columns: Dict[Any, Any] = {}

    if is_scalar(to_replace) and is_scalar(value):
        to_replace_columns = {col: [to_replace] for col in columns_dtype_map}
        values_columns = {col: [value] for col in columns_dtype_map}
    elif cudf.utils.dtypes.is_list_like(to_replace) or isinstance(
        to_replace, cudf.core.column.ColumnBase
    ):
        if is_scalar(value):
            to_replace_columns = {col: to_replace for col in columns_dtype_map}
            values_columns = {
                col: [value]
                if _is_non_decimal_numeric_dtype(columns_dtype_map[col])
                else cudf.utils.utils.scalar_broadcast_to(
                    value, (len(to_replace),), np.dtype(type(value)),
                )
                for col in columns_dtype_map
            }
        elif cudf.utils.dtypes.is_list_like(value):
            if len(to_replace) != len(value):
                raise ValueError(
                    f"Replacement lists must be "
                    f"of same length."
                    f" Expected {len(to_replace)}, got {len(value)}."
                )
            else:
                to_replace_columns = {
                    col: to_replace for col in columns_dtype_map
                }
                values_columns = {col: value for col in columns_dtype_map}
        elif cudf.utils.dtypes.is_column_like(value):
            to_replace_columns = {col: to_replace for col in columns_dtype_map}
            values_columns = {col: value for col in columns_dtype_map}
        else:
            raise TypeError(
                "value argument must be scalar, list-like or Series"
            )
    elif _is_series(to_replace):
        if value is None:
            to_replace_columns = {
                col: as_column(to_replace.index) for col in columns_dtype_map
            }
            values_columns = {col: to_replace for col in columns_dtype_map}
        elif is_dict_like(value):
            to_replace_columns = {
                col: to_replace[col]
                for col in columns_dtype_map
                if col in to_replace
            }
            values_columns = {
                col: value[col] for col in to_replace_columns if col in value
            }
        elif is_scalar(value) or _is_series(value):
            to_replace_columns = {
                col: to_replace[col]
                for col in columns_dtype_map
                if col in to_replace
            }
            values_columns = {
                col: [value] if is_scalar(value) else value[col]
                for col in to_replace_columns
                if col in value
            }
        else:
            raise ValueError(
                "Series.replace cannot use dict-like to_replace and non-None "
                "value"
            )
    elif is_dict_like(to_replace):
        if value is None:
            to_replace_columns = {
                col: list(to_replace.keys()) for col in columns_dtype_map
            }
            values_columns = {
                col: list(to_replace.values()) for col in columns_dtype_map
            }
        elif is_dict_like(value):
            to_replace_columns = {
                col: to_replace[col]
                for col in columns_dtype_map
                if col in to_replace
            }
            values_columns = {
                col: value[col] for col in columns_dtype_map if col in value
            }
        elif is_scalar(value) or _is_series(value):
            to_replace_columns = {
                col: to_replace[col]
                for col in columns_dtype_map
                if col in to_replace
            }
            values_columns = {
                col: [value] if is_scalar(value) else value
                for col in columns_dtype_map
                if col in to_replace
            }
        else:
            raise TypeError("value argument must be scalar, dict, or Series")
    else:
        raise TypeError(
            "Expecting 'to_replace' to be either a scalar, array-like, "
            "dict or None, got invalid type "
            f"'{type(to_replace).__name__}'"
        )

    to_replace_columns = {
        key: [value] if is_scalar(value) else value
        for key, value in to_replace_columns.items()
    }
    values_columns = {
        key: [value] if is_scalar(value) else value
        for key, value in values_columns.items()
    }

    for i in to_replace_columns:
        if i in values_columns:
            if isinstance(values_columns[i], list):
                all_na = values_columns[i].count(None) == len(
                    values_columns[i]
                )
            else:
                all_na = False
            all_na_columns[i] = all_na

    return all_na_columns, to_replace_columns, values_columns


# Create a dictionary of the common, non-null columns
def _get_non_null_cols_and_dtypes(col_idxs, list_of_columns):
    # A mapping of {idx: np.dtype}
    dtypes = dict()
    # A mapping of {idx: [...columns]}, where `[...columns]`
    # is a list of columns with at least one valid value for each
    # column name across all input frames
    non_null_columns = dict()
    for idx in col_idxs:
        for cols in list_of_columns:
            # Skip columns not in this frame
            if idx >= len(cols) or cols[idx] is None:
                continue
            # Store the first dtype we find for a column, even if it's
            # all-null. This ensures we always have at least one dtype
            # for each name. This dtype will be overwritten later if a
            # non-null Column with the same name is found.
            if idx not in dtypes:
                dtypes[idx] = cols[idx].dtype
            if cols[idx].valid_count > 0:
                if idx not in non_null_columns:
                    non_null_columns[idx] = [cols[idx]]
                else:
                    non_null_columns[idx].append(cols[idx])
    return non_null_columns, dtypes


def _find_common_dtypes_and_categories(non_null_columns, dtypes):
    # A mapping of {idx: categories}, where `categories` is a
    # column of all the unique categorical values from each
    # categorical column across all input frames
    categories = dict()
    for idx, cols in non_null_columns.items():
        # default to the first non-null dtype
        dtypes[idx] = cols[0].dtype
        # If all the non-null dtypes are int/float, find a common dtype
        if all(is_numerical_dtype(col.dtype) for col in cols):
            dtypes[idx] = find_common_type([col.dtype for col in cols])
        # If all categorical dtypes, combine the categories
        elif all(
            isinstance(col, cudf.core.column.CategoricalColumn) for col in cols
        ):
            # Combine and de-dupe the categories
            categories[idx] = (
                cudf.concat([col.cat().categories for col in cols])
                .to_series()
                .drop_duplicates(ignore_index=True)
                ._column
            )
            # Set the column dtype to the codes' dtype. The categories
            # will be re-assigned at the end
            dtypes[idx] = min_scalar_type(len(categories[idx]))
        # Otherwise raise an error if columns have different dtypes
        elif not all(is_dtype_equal(c.dtype, dtypes[idx]) for c in cols):
            raise ValueError("All columns must be the same type")
    return categories


def _cast_cols_to_common_dtypes(col_idxs, list_of_columns, dtypes, categories):
    # Cast all columns to a common dtype, assign combined categories,
    # and back-fill missing columns with all-null columns
    for idx in col_idxs:
        dtype = dtypes[idx]
        for cols in list_of_columns:
            # If column not in this df, fill with an all-null column
            if idx >= len(cols) or cols[idx] is None:
                n = len(next(x for x in cols if x is not None))
                cols[idx] = column_empty(row_count=n, dtype=dtype, masked=True)
            else:
                # If column is categorical, rebase the codes with the
                # combined categories, and cast the new codes to the
                # min-scalar-sized dtype
                if idx in categories:
                    cols[idx] = (
                        cols[idx]
                        .cat()
                        ._set_categories(
                            cols[idx].cat().categories,
                            categories[idx],
                            is_unique=True,
                        )
                        .codes
                    )
                cols[idx] = cols[idx].astype(dtype)


def _reassign_categories(categories, cols, col_idxs):
    for name, idx in zip(cols, col_idxs):
        if idx in categories:
            cols[name] = build_categorical_column(
                categories=categories[idx],
                codes=as_column(cols[name].base_data, dtype=cols[name].dtype),
                mask=cols[name].base_mask,
                offset=cols[name].offset,
                size=cols[name].size,
            )


def _is_series(obj):
    """
    Checks if the `obj` is of type `cudf.Series`
    instead of checking for isinstance(obj, cudf.Series)
    """
    return isinstance(obj, Frame) and obj.ndim == 1 and obj._index is not None


def _drop_rows_by_labels(
    obj: DataFrameOrSeries,
    labels: Union[ColumnLike, abc.Iterable, str],
    level: Union[int, str],
    errors: str,
) -> DataFrameOrSeries:
    """Remove rows specified by `labels`. If `errors=True`, an error is raised
    if some items in `labels` do not exist in `obj._index`.

    Will raise if level(int) is greater or equal to index nlevels
    """
    if isinstance(level, int) and level >= obj.index.nlevels:
        raise ValueError("Param level out of bounds.")

    if not isinstance(labels, SingleColumnFrame):
        labels = as_column(labels)

    if isinstance(obj._index, cudf.MultiIndex):
        if level is None:
            level = 0

        levels_index = obj.index.get_level_values(level)
        if errors == "raise" and not labels.isin(levels_index).all():
            raise KeyError("One or more values not found in axis")

        if isinstance(level, int):
            ilevel = level
        else:
            ilevel = obj._index.names.index(level)

        # 1. Merge Index df and data df along column axis:
        # | id | ._index df | data column(s) |
        idx_nlv = obj._index.nlevels
        working_df = obj._index._source_data
        working_df.columns = [i for i in range(idx_nlv)]
        for i, col in enumerate(obj._data):
            working_df[idx_nlv + i] = obj._data[col]
        # 2. Set `level` as common index:
        # | level | ._index df w/o level | data column(s) |
        working_df = working_df.set_index(level)

        # 3. Use "leftanti" join to drop
        # TODO: use internal API with "leftanti" and specify left and right
        # join keys to bypass logic check
        to_join = cudf.DataFrame(index=cudf.Index(labels, name=level))
        join_res = working_df.join(to_join, how="leftanti")

        # 4. Reconstruct original layout, and rename
        join_res.insert(
            ilevel, name=join_res._index.name, value=join_res._index
        )
        join_res = join_res.reset_index(drop=True)

        midx = cudf.MultiIndex.from_frame(
            join_res.iloc[:, 0:idx_nlv], names=obj._index.names
        )

        if isinstance(obj, cudf.Series):
            return obj.__class__._from_data(
                join_res.iloc[:, idx_nlv:]._data, index=midx, name=obj.name
            )
        else:
            return obj.__class__._from_data(
                join_res.iloc[:, idx_nlv:]._data,
                index=midx,
                columns=obj.columns,
            )

    else:
        if errors == "raise" and not labels.isin(obj.index).all():
            raise KeyError("One or more values not found in axis")

        key_df = cudf.DataFrame(index=labels)
        if isinstance(obj, cudf.Series):
            res = obj.to_frame(name="tmp").join(key_df, how="leftanti")["tmp"]
            res.name = obj.name
            return res
        else:
            return obj.join(key_df, how="leftanti")
