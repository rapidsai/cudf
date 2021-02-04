# Copyright (c) 2020-2021, NVIDIA CORPORATION.
from __future__ import annotations

import copy
import functools
import operator
import warnings
from collections import OrderedDict, abc as abc
from typing import TYPE_CHECKING, Any, Dict, Tuple, overload

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
from nvtx import annotate
from pandas.api.types import is_dict_like, is_dtype_equal
from typing_extensions import Literal

import cudf
from cudf import _lib as libcudf
from cudf.core.column import as_column, build_categorical_column, column_empty
from cudf.utils.dtypes import (
    is_categorical_dtype,
    is_column_like,
    is_numerical_dtype,
    is_scalar,
    min_scalar_type,
)

if TYPE_CHECKING:
    from cudf.core.column_accessor import ColumnAccessor


class Frame(libcudf.table.Table):
    """
    Frame: A collection of Column objects with an optional index.

    Parameters
    ----------
    data : OrderedColumnDict
        An OrderedColumnDict mapping column names to Columns
    index : Table
        A Frame representing the (optional) index columns.
    """

    _data: "ColumnAccessor"

    @classmethod
    def _from_table(cls, table: Frame):
        return cls(table._data, index=table._index)

    @overload
    def _mimic_inplace(self, result: Frame) -> Frame:
        ...

    @overload
    def _mimic_inplace(self, result: Frame, inplace: Literal[True]):
        ...

    @overload
    def _mimic_inplace(self, result: Frame, inplace: Literal[False]) -> Frame:
        ...

    def _mimic_inplace(self, result, inplace=False):
        if inplace:
            for col in self._data:
                if col in result._data:
                    self._data[col]._mimic_inplace(
                        result._data[col], inplace=True
                    )
            self._data = result._data
            self._index = result._index
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

    def copy(self, deep=True):
        return Frame(self._data.copy(deep=deep))

    @classmethod
    @annotate("CONCAT", color="orange", domain="cudf_python")
    def _concat(
        cls, objs, axis=0, join="outer", ignore_index=False, sort=False
    ):
        # shallow-copy the input DFs in case the same DF instance
        # is concatenated with itself

        # flag to indicate at least one empty input frame also has an index
        empty_has_index = False
        # length of output frame's RangeIndex if all input frames are empty,
        # and at least one has an index
        result_index_length = 0
        # the number of empty input frames
        num_empty_input_frames = 0

        for i, obj in enumerate(objs):
            objs[i] = obj.copy(deep=False)
            if ignore_index:
                # If ignore_index is true, determine if
                # all or some objs are empty(and have index).
                # 1. If all objects are empty(and have index), we
                # should set the index separately using RangeIndex.
                # 2. If some objects are empty(and have index), we
                # create empty columns later while populating `columns`
                # variable. Detailed explanation of second case before
                # allocation of `columns` variable below.
                if obj.empty:
                    num_empty_input_frames += 1
                    result_index_length += len(obj)
                    empty_has_index = empty_has_index or len(obj) > 0

        if join == "inner":
            all_columns_list = [obj._column_names for obj in objs]
            # get column names present in ALL objs
            intersecting_columns = functools.reduce(
                np.intersect1d, all_columns_list
            )
            # get column names not present in all objs
            non_intersecting_columns = (
                functools.reduce(operator.or_, (obj.columns for obj in objs))
                ^ intersecting_columns
            )
            names = OrderedDict.fromkeys(intersecting_columns).keys()

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

                objs = [obj.copy(deep=False) for obj in objs]
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
            names = OrderedDict.fromkeys(names).keys()

        else:
            raise ValueError(
                "Only can inner (intersect) or outer (union) when joining"
                "the other axis"
            )

        try:
            if sort:
                names = list(sorted(names))
            else:
                names = list(names)
        except TypeError:
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
            for i, f in enumerate(objs)
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

    def _get_columns_by_label(self, labels, downcast):
        """
        Returns columns of the Frame specified by `labels`

        """
        return self._data.select_by_label(labels)

    def _get_columns_by_index(self, indices):
        """
        Returns columns of the Frame specified by `labels`

        """
        data = self._data.select_by_index(indices)
        return self._constructor(
            data, columns=data.to_pandas_index(), index=self.index
        )

    def _gather(self, gather_map, keep_index=True):
        if not pd.api.types.is_integer_dtype(gather_map.dtype):
            gather_map = gather_map.astype("int32")
        result = self.__class__._from_table(
            libcudf.copying.gather(
                self, as_column(gather_map), keep_index=keep_index
            )
        )
        result._copy_type_metadata(self)
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

    def _slice(self, arg):
        """
       _slice : slice the frame as per the arg

       Parameters
       ----------
       arg : should always be of type slice and doesn't handle step

       """
        from cudf.core.index import RangeIndex

        num_rows = len(self)
        if num_rows == 0:
            return self
        start, stop, stride = arg.indices(num_rows)

        # This is just to handle RangeIndex type, stop
        # it from materializing unnecessarily
        keep_index = True
        if self.index is not None and isinstance(self.index, RangeIndex):
            if self._num_columns == 0:
                result = self._empty_like(keep_index)
                result._index = self.index[start:stop]
                return result
            keep_index = False

        if start < 0:
            start = start + num_rows
        if stop < 0:
            stop = stop + num_rows

        if start > stop and (stride is None or stride == 1):
            return self._empty_like(keep_index)
        else:
            start = len(self) if start > num_rows else start
            stop = len(self) if stop > num_rows else stop

            if stride is not None and stride != 1:
                return self._gather(
                    cudf.core.column.arange(
                        start, stop=stop, step=stride, dtype=np.int32
                    )
                )
            else:
                result = self._from_table(
                    libcudf.copying.table_slice(
                        self, [start, stop], keep_index
                    )[0]
                )

                result._copy_type_metadata(self, include_index=keep_index)
                # Adding index of type RangeIndex back to
                # result
                if keep_index is False and self.index is not None:
                    result.index = self.index[start:stop]
                result.columns = self.columns
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

    def _normalize_scalars(self, other):
        """
        Try to normalizes scalar values as per self dtype
        """
        if (
            other is not None
            and (isinstance(other, float) and not np.isnan(other))
        ) and (self.dtype.type(other) != other):
            raise TypeError(
                f"Cannot safely cast non-equivalent "
                f"{type(other).__name__} to {self.dtype.name}"
            )

        return (
            self.dtype.type(other)
            if (
                other is not None
                and (isinstance(other, float) and not np.isnan(other))
            )
            else other
        )

    def _normalize_columns_and_scalars_type(self, other):
        """
        Try to normalize the other's dtypes as per self.

        Parameters
        ----------

        self : Can be a DataFrame or Series or Index
        other : Can be a DataFrame, Series, Index, Array
            like object or a scalar value

            if self is DataFrame, other can be only a
            scalar or array like with size of number of columns
            in DataFrame or a DataFrame with same dimension

            if self is Series, other can be only a scalar or
            a series like with same length as self

        Returns:
        --------
        A dataframe/series/list/scalar form of normalized other
        """
        if isinstance(self, cudf.DataFrame) and isinstance(
            other, cudf.DataFrame
        ):
            return [
                other[self_col].astype(self._data[self_col].dtype)._column
                for self_col in self._data.names
            ]

        elif isinstance(self, (cudf.Series, cudf.Index)) and not is_scalar(
            other
        ):
            other = as_column(other)
            return other.astype(self.dtype)

        else:
            # Handles scalar or list/array like scalars
            if isinstance(self, (cudf.Series, cudf.Index)) and is_scalar(
                other
            ):
                return self._normalize_scalars(other)

            elif isinstance(self, cudf.DataFrame):
                out = []
                if is_scalar(other):
                    other = [other for i in range(len(self._data.names))]
                out = [
                    self[in_col_name]._normalize_scalars(sclr)
                    for in_col_name, sclr in zip(self._data.names, other)
                ]

                return out
            else:
                raise ValueError(
                    f"Inappropriate input {type(self)} "
                    f"and other {type(other)} combination"
                )

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

        if isinstance(self, cudf.DataFrame):
            if hasattr(cond, "__cuda_array_interface__"):
                cond = cudf.DataFrame(
                    cond, columns=self._data.names, index=self.index
                )
            elif not isinstance(cond, cudf.DataFrame):
                cond = self.from_pandas(pd.DataFrame(cond))

            common_cols = set(self._data.names).intersection(
                set(cond._data.names)
            )
            if len(common_cols) > 0:
                # If `self` and `cond` are having unequal index,
                # then re-index `cond`.
                if not self.index.equals(cond.index):
                    cond = cond.reindex(self.index)
            else:
                if cond.shape != self.shape:
                    raise ValueError(
                        """Array conditional must be same shape as self"""
                    )
                # Setting `self` column names to `cond`
                # as `cond` has no column names.
                cond.columns = self.columns

            other = self._normalize_columns_and_scalars_type(other)
            out_df = cudf.DataFrame(index=self.index)
            if len(self._columns) != len(other):
                raise ValueError(
                    """Replacement list length or number of dataframe columns
                    should be equal to Number of columns of dataframe"""
                )

            for column_name, other_column in zip(self._data.names, other):
                input_col = self._data[column_name]
                if column_name in cond._data:
                    if isinstance(
                        input_col, cudf.core.column.CategoricalColumn
                    ):
                        if np.isscalar(other_column):
                            try:
                                other_column = input_col._encode(other_column)
                            except ValueError:
                                # When other is not present in categories,
                                # fill with Null.
                                other_column = None
                        elif hasattr(other_column, "codes"):
                            other_column = other_column.codes
                        input_col = input_col.codes

                    result = libcudf.copying.copy_if_else(
                        input_col, other_column, cond._data[column_name]
                    )

                    if isinstance(
                        self._data[column_name],
                        cudf.core.column.CategoricalColumn,
                    ):
                        result = build_categorical_column(
                            categories=self._data[column_name].categories,
                            codes=as_column(
                                result.base_data, dtype=result.dtype
                            ),
                            mask=result.base_mask,
                            size=result.size,
                            offset=result.offset,
                            ordered=self._data[column_name].ordered,
                        )
                else:
                    from cudf._lib.null_mask import MaskState, create_null_mask

                    out_mask = create_null_mask(
                        len(input_col), state=MaskState.ALL_NULL
                    )
                    result = input_col.set_mask(out_mask)
                out_df[column_name] = self[column_name].__class__(result)

            return self._mimic_inplace(out_df, inplace=inplace)

        else:

            if isinstance(other, cudf.DataFrame):
                raise NotImplementedError(
                    "cannot align with a higher dimensional Frame"
                )

            other = self._normalize_columns_and_scalars_type(other)

            cond = as_column(cond)
            if len(cond) != len(self):
                raise ValueError(
                    """Array conditional must be same shape as self"""
                )
            input_col = self._data[self.name]
            if isinstance(input_col, cudf.core.column.CategoricalColumn):
                if np.isscalar(other):
                    try:
                        other = input_col._encode(other)
                    except ValueError:
                        # When other is not present in categories,
                        # fill with Null.
                        other = None
                elif hasattr(other, "codes"):
                    other = other.codes

                input_col = input_col.codes

            result = libcudf.copying.copy_if_else(input_col, other, cond)

            if is_categorical_dtype(self.dtype):
                result = build_categorical_column(
                    categories=self._data[self.name].categories,
                    codes=as_column(result.base_data, dtype=result.dtype),
                    mask=result.base_mask,
                    size=result.size,
                    offset=result.offset,
                    ordered=self._data[self.name].ordered,
                )

            if isinstance(self, cudf.Index):
                from cudf.core.index import as_index

                result = as_index(result, name=self.name)
            else:
                result = self._copy_construct(data=result)

            return self._mimic_inplace(result, inplace=inplace)

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
        if map_index.dtype == np.float:
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
        if boolean_mask.has_nulls:
            raise ValueError(
                "cannot mask with boolean_mask containing null values"
            )
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
        axis : {0 or 'index', 1 or 'columns'}, default 0
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

        source = self
        if numeric_only:
            numeric_cols = (
                name
                for name in self._data.names
                if is_numerical_dtype(self._data[name])
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

    def round(self, decimals=0):
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
                name: col.round(decimals[name])
                if (
                    name in decimals.keys()
                    and pd.api.types.is_numeric_dtype(col.dtype)
                )
                else col.copy(deep=True)
                for name, col in self._data.items()
            }
        elif isinstance(decimals, int):
            cols = {
                name: col.round(decimals)
                if pd.api.types.is_numeric_dtype(col.dtype)
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
                except KeyError:
                    # We need to create a deep copy if `find_and_replace`
                    # was not successful or any of
                    # `to_replace_per_column`, `replacements_per_column`,
                    # `all_na_per_column` don't contain the `name`
                    # that exists in `copy_data`
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
        See `ColumnBase._copy_type_metadata` for more information.
        """
        for name, col, other_col in zip(
            self._data.keys(), self._data.values(), other._data.values()
        ):
            self._data[name] = other_col._copy_type_metadata(col)

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
        from cudf.utils.dtypes import is_scalar

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

    def _get_sorted_inds(self, ascending=True, na_position="last"):
        """
        Sort by the values.

        Parameters
        ----------
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

        # If given a scalar need to construct a sequence of length # of columns
        if np.isscalar(ascending):
            ascending = [ascending] * self._num_columns

        return libcudf.sort.order_by(self, ascending, na_position)

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

    @staticmethod
    def _validate_merge_cfg(
        lhs,
        rhs,
        left_on,
        right_on,
        on,
        how,
        left_index=False,
        right_index=False,
        lsuffix=None,
        rsuffix=None,
    ):
        """
        Error for various combinations of merge input parameters
        """
        len_left_on = len(left_on) if left_on is not None else 0
        len_right_on = len(right_on) if right_on is not None else 0

        # must actually support the requested merge type
        if how not in ["left", "inner", "outer", "leftanti", "leftsemi"]:
            raise NotImplementedError(f"{how} merge not supported yet")

        # Passing 'on' with 'left_on' or 'right_on' is potentially ambiguous
        if on:
            if left_on or right_on:
                raise ValueError(
                    'Can only pass argument "on" OR "left_on" '
                    'and "right_on", not a combination of both.'
                )

        # Require same total number of columns to join on in both operands
        if not (len_left_on + left_index * len(lhs.index.names)) == (
            len_right_on + right_index * len(rhs.index.names)
        ):
            raise ValueError(
                "Merge operands must have same number of join key columns"
            )

        # If nothing specified, must have common cols to use implicitly
        same_named_columns = set(lhs._data.keys()) & set(rhs._data.keys())
        if not (left_index or right_index):
            if not (left_on or right_on):
                if len(same_named_columns) == 0:
                    raise ValueError("No common columns to perform merge on")

        for name in same_named_columns:
            if not (
                name in left_on
                and name in right_on
                and (left_on.index(name) == right_on.index(name))
            ):
                if not (lsuffix or rsuffix):
                    raise ValueError(
                        "there are overlapping columns but "
                        "lsuffix and rsuffix are not defined"
                    )

        if on:
            on_keys = [on] if not isinstance(on, list) else on
            for key in on_keys:
                if not (key in lhs._data.keys() and key in rhs._data.keys()):
                    raise KeyError(f"Key {on} not in both operands")
        else:
            for key in left_on:
                if key not in lhs._data.keys():
                    raise KeyError(f'Key "{key}" not in left operand')
            for key in right_on:
                if key not in rhs._data.keys():
                    raise KeyError(f'Key "{key}" not in right operand')

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
        lsuffix=None,
        rsuffix=None,
        method="hash",
        indicator=False,
        suffixes=("_x", "_y"),
    ):
        # Merge doesn't support right, so just swap
        if how == "right":
            return right._merge(
                self,
                on=on,
                left_on=right_on,
                right_on=left_on,
                left_index=right_index,
                right_index=left_index,
                how="left",
                sort=sort,
                lsuffix=rsuffix,
                rsuffix=lsuffix,
                method=method,
                indicator=indicator,
                suffixes=suffixes,
            )

        lhs = self
        rhs = right

        from cudf.core.join import Merge

        mergeop = Merge(
            lhs,
            rhs,
            on,
            left_on,
            right_on,
            left_index,
            right_index,
            how,
            sort,
            lsuffix,
            rsuffix,
            method,
            indicator,
            suffixes,
        )
        to_return = mergeop.perform_merge()

        # If sort=True, Pandas would sort on the key columns in the
        # same order as given in 'on'. If the indices are used as
        # keys, the index will be sorted. If one index is specified,
        # the key column on the other side will be used to sort.
        # If no index is specified, return a new RangeIndex
        if sort:
            to_sort = cudf.DataFrame()
            if left_index and right_index:
                by = list(to_return._index._data.columns)
                if left_on and right_on:
                    by.extend(to_return[mergeop.left_on]._data.columns)
            elif left_index:
                by = list(to_return[mergeop.right_on]._data.columns)
            elif right_index:
                by = list(to_return[mergeop.left_on]._data.columns)
            else:
                # left_on == right_on, or different names but same columns
                # in both cases we can sort by either
                by = [to_return._data[name] for name in mergeop.left_on]
            for i, col in enumerate(by):
                to_sort[i] = col
            inds = to_sort.argsort()
            if isinstance(to_return, cudf.Index):
                to_return = to_return.take(inds)
            else:
                to_return = to_return.take(
                    inds, keep_index=(left_index or right_index)
                )
            return to_return
        else:
            return to_return

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

        cols = OrderedDict()
        index = index if index is not None else df.index
        names = columns if columns is not None else list(df.columns)
        for name in names:
            if name in df._data:
                cols[name] = df._data[name].copy(deep=deep)
            else:
                dtype = dtypes.get(name, np.float64)
                cols[name] = column_empty(
                    dtype=dtype, masked=True, row_count=len(index)
                )
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
                if pd.api.types.is_numeric_dtype(columns_dtype_map[col])
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


# If the dictionary array is a string array and of length `0`
# it should be a null array
def _get_dictionary_array(array):
    if isinstance(array, pa.StringArray) and len(array) == 0:
        return pa.array([], type=pa.null())
    else:
        return array


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
            dtypes[idx] = np.find_common_type([col.dtype for col in cols], [])
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
