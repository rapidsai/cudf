# Copyright (c) 2018-2021, NVIDIA CORPORATION.

from __future__ import annotations, division

import functools
import inspect
import itertools
import numbers
import pickle
import sys
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, MutableMapping, Optional, Set, TypeVar

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
from numba import cuda
from nvtx import annotate
from pandas._config import get_option
from pandas.io.formats import console
from pandas.io.formats.printing import pprint_thing

import cudf
import cudf.core.common
from cudf import _lib as libcudf
from cudf.api.types import (
    _is_scalar_or_zero_d_array,
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime_dtype,
    is_dict_like,
    is_dtype_equal,
    is_list_dtype,
    is_list_like,
    is_numeric_dtype,
    is_scalar,
    is_string_dtype,
    is_struct_dtype,
)
from cudf.core import column, df_protocol, reshape
from cudf.core.abc import Serializable
from cudf.core.column import (
    as_column,
    build_categorical_column,
    build_column,
    column_empty,
    concat_columns,
)
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.frame import Frame, _drop_rows_by_labels
from cudf.core.groupby.groupby import DataFrameGroupBy
from cudf.core.index import BaseIndex, RangeIndex, as_index
from cudf.core.indexed_frame import (
    IndexedFrame,
    _FrameIndexer,
    _get_label_range_or_mask,
    _indices_from_labels,
    doc_reset_index_template,
)
from cudf.core.multiindex import MultiIndex
from cudf.core.resample import DataFrameResampler
from cudf.core.series import Series
from cudf.utils import applyutils, docutils, ioutils, queryutils, utils
from cudf.utils.docutils import copy_docstring
from cudf.utils.dtypes import (
    can_convert_to_column,
    cudf_dtype_from_pydata_dtype,
    find_common_type,
    is_column_like,
    min_scalar_type,
    numeric_normalize_types,
)
from cudf.utils.utils import GetAttrGetItemMixin

T = TypeVar("T", bound="DataFrame")


_cupy_nan_methods_map = {
    "min": "nanmin",
    "max": "nanmax",
    "sum": "nansum",
    "prod": "nanprod",
    "product": "nanprod",
    "mean": "nanmean",
    "std": "nanstd",
    "var": "nanvar",
}


class _DataFrameIndexer(_FrameIndexer):
    def __getitem__(self, arg):
        if isinstance(self._frame.index, MultiIndex) or isinstance(
            self._frame.columns, MultiIndex
        ):
            # This try/except block allows the use of pandas-like
            # tuple arguments into MultiIndex dataframes.
            try:
                return self._getitem_tuple_arg(arg)
            except (TypeError, KeyError, IndexError, ValueError):
                return self._getitem_tuple_arg((arg, slice(None)))
        else:
            if not isinstance(arg, tuple):
                arg = (arg, slice(None))
            return self._getitem_tuple_arg(arg)

    def __setitem__(self, key, value):
        if not isinstance(key, tuple):
            key = (key, slice(None))
        return self._setitem_tuple_arg(key, value)

    def _can_downcast_to_series(self, df, arg):
        """
        This method encapsulates the logic used
        to determine whether or not the result of a loc/iloc
        operation should be "downcasted" from a DataFrame to a
        Series
        """
        if isinstance(df, cudf.Series):
            return False
        nrows, ncols = df.shape
        if nrows == 1:
            if type(arg[0]) is slice:
                if not is_scalar(arg[1]):
                    return False
            elif (is_list_like(arg[0]) or is_column_like(arg[0])) and (
                is_list_like(arg[1])
                or is_column_like(arg[0])
                or type(arg[1]) is slice
            ):
                return False
            else:
                if is_bool_dtype(as_column(arg[0]).dtype) and not isinstance(
                    arg[1], slice
                ):
                    return True
            dtypes = df.dtypes.values.tolist()
            all_numeric = all([is_numeric_dtype(t) for t in dtypes])
            if all_numeric:
                return True
        if ncols == 1:
            if type(arg[1]) is slice:
                return False
            if isinstance(arg[1], tuple):
                # Multiindex indexing with a slice
                if any(isinstance(v, slice) for v in arg):
                    return False
            if not (is_list_like(arg[1]) or is_column_like(arg[1])):
                return True
        return False

    def _downcast_to_series(self, df, arg):
        """
        "Downcast" from a DataFrame to a Series
        based on Pandas indexing rules
        """
        nrows, ncols = df.shape
        # determine the axis along which the Series is taken:
        if nrows == 1 and ncols == 1:
            if is_scalar(arg[0]) and is_scalar(arg[1]):
                return df[df.columns[0]].iloc[0]
            elif not is_scalar(arg[0]):
                axis = 1
            else:
                axis = 0

        elif nrows == 1:
            axis = 0
        elif ncols == 1:
            axis = 1
        else:
            raise ValueError("Cannot downcast DataFrame selection to Series")

        # take series along the axis:
        if axis == 1:
            return df[df._data.names[0]]
        else:
            if df._num_columns > 0:
                dtypes = df.dtypes.values.tolist()
                normalized_dtype = np.result_type(*dtypes)
                for name, col in df._data.items():
                    df[name] = col.astype(normalized_dtype)

            sr = df.T
            return sr[sr._data.names[0]]


class _DataFrameLocIndexer(_DataFrameIndexer):
    """
    For selection by label.
    """

    def _getitem_scalar(self, arg):
        return self._frame[arg[1]].loc[arg[0]]

    @annotate("LOC_GETITEM", color="blue", domain="cudf_python")
    def _getitem_tuple_arg(self, arg):
        from uuid import uuid4

        # Step 1: Gather columns
        if isinstance(arg, tuple):
            columns_df = self._frame._get_columns_by_label(arg[1])
            columns_df._index = self._frame._index
        else:
            columns_df = self._frame

        # Step 2: Gather rows
        if isinstance(columns_df.index, MultiIndex):
            if isinstance(arg, (MultiIndex, pd.MultiIndex)):
                if isinstance(arg, pd.MultiIndex):
                    arg = MultiIndex.from_pandas(arg)

                indices = _indices_from_labels(columns_df, arg)
                return columns_df.take(indices)

            else:
                if isinstance(arg, tuple):
                    return columns_df.index._get_row_major(columns_df, arg[0])
                else:
                    return columns_df.index._get_row_major(columns_df, arg)
        else:
            if isinstance(arg[0], slice):
                out = _get_label_range_or_mask(
                    columns_df.index, arg[0].start, arg[0].stop, arg[0].step
                )
                if isinstance(out, slice):
                    df = columns_df._slice(out)
                else:
                    df = columns_df._apply_boolean_mask(out)
            else:
                tmp_arg = arg
                if is_scalar(arg[0]):
                    # If a scalar, there is possibility of having duplicates.
                    # Join would get all the duplicates. So, coverting it to
                    # an array kind.
                    tmp_arg = ([tmp_arg[0]], tmp_arg[1])
                if len(tmp_arg[0]) == 0:
                    return columns_df._empty_like(keep_index=True)
                tmp_arg = (as_column(tmp_arg[0]), tmp_arg[1])

                if is_bool_dtype(tmp_arg[0]):
                    df = columns_df._apply_boolean_mask(tmp_arg[0])
                else:
                    tmp_col_name = str(uuid4())
                    other_df = DataFrame(
                        {tmp_col_name: column.arange(len(tmp_arg[0]))},
                        index=as_index(tmp_arg[0]),
                    )
                    df = other_df.join(columns_df, how="inner")
                    # as join is not assigning any names to index,
                    # update it over here
                    df.index.name = columns_df.index.name
                    df = df.sort_values(tmp_col_name)
                    df.drop(columns=[tmp_col_name], inplace=True)
                    # There were no indices found
                    if len(df) == 0:
                        raise KeyError(arg)

        # Step 3: Gather index
        if df.shape[0] == 1:  # we have a single row
            if isinstance(arg[0], slice):
                start = arg[0].start
                if start is None:
                    start = self._frame.index[0]
                df.index = as_index(start)
            else:
                row_selection = as_column(arg[0])
                if is_bool_dtype(row_selection.dtype):
                    df.index = self._frame.index.take(row_selection)
                else:
                    df.index = as_index(row_selection)
        # Step 4: Downcast
        if self._can_downcast_to_series(df, arg):
            return self._downcast_to_series(df, arg)
        return df

    @annotate("LOC_SETITEM", color="blue", domain="cudf_python")
    def _setitem_tuple_arg(self, key, value):
        if isinstance(self._frame.index, MultiIndex) or isinstance(
            self._frame.columns, pd.MultiIndex
        ):
            raise NotImplementedError(
                "Setting values using df.loc[] not supported on "
                "DataFrames with a MultiIndex"
            )

        try:
            columns_df = self._frame._get_columns_by_label(key[1])
        except KeyError:
            if not self._frame.empty and isinstance(key[0], slice):
                pos_range = _get_label_range_or_mask(
                    self._frame.index, key[0].start, key[0].stop, key[0].step
                )
                idx = self._frame.index[pos_range]
            elif self._frame.empty and isinstance(key[0], slice):
                idx = None
            else:
                idx = cudf.Index(key[0])
            if is_scalar(value):
                length = len(idx) if idx is not None else 1
                value = as_column(value, length=length)

            new_col = cudf.Series(value, index=idx)
            if not self._frame.empty:
                new_col = new_col._align_to_index(
                    self._frame.index, how="right"
                )

            if self._frame.empty:
                self._frame.index = (
                    idx if idx is not None else cudf.RangeIndex(len(new_col))
                )
            self._frame._data.insert(key[1], new_col)
        else:
            if isinstance(value, (cupy.ndarray, np.ndarray)):
                value_df = DataFrame(value)
                if value_df.shape[1] != columns_df.shape[1]:
                    if value_df.shape[1] == 1:
                        value_cols = (
                            value_df._data.columns * columns_df.shape[1]
                        )
                    else:
                        raise ValueError(
                            f"shape mismatch: value array of shape "
                            f"{value_df.shape} could not be "
                            f"broadcast to indexing result of shape "
                            f"{columns_df.shape}"
                        )
                else:
                    value_cols = value_df._data.columns
                for i, col in enumerate(columns_df._column_names):
                    self._frame[col].loc[key[0]] = value_cols[i]
            else:
                for col in columns_df._column_names:
                    self._frame[col].loc[key[0]] = value


class _DataFrameIlocIndexer(_DataFrameIndexer):
    """
    For selection by index.
    """

    @annotate("ILOC_GETITEM", color="blue", domain="cudf_python")
    def _getitem_tuple_arg(self, arg):
        # Iloc Step 1:
        # Gather the columns specified by the second tuple arg
        columns_df = DataFrame(self._frame._get_columns_by_index(arg[1]))

        columns_df._index = self._frame._index

        # Iloc Step 2:
        # Gather the rows specified by the first tuple arg
        if isinstance(columns_df.index, MultiIndex):
            if isinstance(arg[0], slice):
                df = columns_df[arg[0]]
            else:
                df = columns_df.index._get_row_major(columns_df, arg[0])
            if (len(df) == 1 and len(columns_df) >= 1) and not (
                isinstance(arg[0], slice) or isinstance(arg[1], slice)
            ):
                # Pandas returns a numpy scalar in this case
                return df.iloc[0]
            if self._can_downcast_to_series(df, arg):
                return self._downcast_to_series(df, arg)
            return df
        else:
            if isinstance(arg[0], slice):
                df = columns_df._slice(arg[0])
            elif is_scalar(arg[0]):
                index = arg[0]
                if index < 0:
                    index += len(columns_df)
                df = columns_df._slice(slice(index, index + 1, 1))
            else:
                arg = (as_column(arg[0]), arg[1])
                if is_bool_dtype(arg[0]):
                    df = columns_df._apply_boolean_mask(arg[0])
                else:
                    df = columns_df._gather(arg[0])

        # Iloc Step 3:
        # Reindex
        if df.shape[0] == 1:  # we have a single row without an index
            df.index = as_index(self._frame.index[arg[0]])

        # Iloc Step 4:
        # Downcast
        if self._can_downcast_to_series(df, arg):
            return self._downcast_to_series(df, arg)

        if df.shape[0] == 0 and df.shape[1] == 0 and isinstance(arg[0], slice):
            df._index = as_index(self._frame.index[arg[0]])
        return df

    @annotate("ILOC_SETITEM", color="blue", domain="cudf_python")
    def _setitem_tuple_arg(self, key, value):
        columns = DataFrame(self._frame._get_columns_by_index(key[1]))

        for col in columns:
            self._frame[col].iloc[key[0]] = value

    def _getitem_scalar(self, arg):
        col = self._frame.columns[arg[1]]
        return self._frame[col].iloc[arg[0]]


class DataFrame(IndexedFrame, Serializable, GetAttrGetItemMixin):
    """
    A GPU Dataframe object.

    Parameters
    ----------
    data : array-like, Iterable, dict, or DataFrame.
        Dict can contain Series, arrays, constants, or list-like objects.

    index : Index or array-like
        Index to use for resulting frame. Will default to
        RangeIndex if no indexing information part of input data and
        no index provided.

    columns : Index or array-like
        Column labels to use for resulting frame.
        Will default to RangeIndex (0, 1, 2, …, n) if no column
        labels are provided.

    dtype : dtype, default None
        Data type to force. Only a single dtype is allowed.
        If None, infer.

    nan_as_null : bool, Default True
        If ``None``/``True``, converts ``np.nan`` values to
        ``null`` values.
        If ``False``, leaves ``np.nan`` values as is.

    Examples
    --------

    Build dataframe with ``__setitem__``:

    >>> import cudf
    >>> df = cudf.DataFrame()
    >>> df['key'] = [0, 1, 2, 3, 4]
    >>> df['val'] = [float(i + 10) for i in range(5)]  # insert column
    >>> df
       key   val
    0    0  10.0
    1    1  11.0
    2    2  12.0
    3    3  13.0
    4    4  14.0

    Build DataFrame via dict of columns:

    >>> import numpy as np
    >>> from datetime import datetime, timedelta
    >>> t0 = datetime.strptime('2018-10-07 12:00:00', '%Y-%m-%d %H:%M:%S')
    >>> n = 5
    >>> df = cudf.DataFrame({
    ...     'id': np.arange(n),
    ...     'datetimes': np.array(
    ...     [(t0+ timedelta(seconds=x)) for x in range(n)])
    ... })
    >>> df
        id                datetimes
    0    0  2018-10-07T12:00:00.000
    1    1  2018-10-07T12:00:01.000
    2    2  2018-10-07T12:00:02.000
    3    3  2018-10-07T12:00:03.000
    4    4  2018-10-07T12:00:04.000

    Build DataFrame via list of rows as tuples:

    >>> df = cudf.DataFrame([
    ...     (5, "cats", "jump", np.nan),
    ...     (2, "dogs", "dig", 7.5),
    ...     (3, "cows", "moo", -2.1, "occasionally"),
    ... ])
    >>> df
       0     1     2     3             4
    0  5  cats  jump  <NA>          <NA>
    1  2  dogs   dig   7.5          <NA>
    2  3  cows   moo  -2.1  occasionally

    Convert from a Pandas DataFrame:

    >>> import pandas as pd
    >>> pdf = pd.DataFrame({'a': [0, 1, 2, 3],'b': [0.1, 0.2, None, 0.3]})
    >>> pdf
       a    b
    0  0  0.1
    1  1  0.2
    2  2  NaN
    3  3  0.3
    >>> df = cudf.from_pandas(pdf)
    >>> df
       a     b
    0  0   0.1
    1  1   0.2
    2  2  <NA>
    3  3   0.3
    """

    _PROTECTED_KEYS = frozenset(("_data", "_index"))
    _accessors: Set[Any] = set()
    _loc_indexer_type = _DataFrameLocIndexer
    _iloc_indexer_type = _DataFrameIlocIndexer

    @annotate("DATAFRAME_INIT", color="blue", domain="cudf_python")
    def __init__(
        self, data=None, index=None, columns=None, dtype=None, nan_as_null=True
    ):

        super().__init__()

        if isinstance(columns, (Series, cudf.BaseIndex)):
            columns = columns.to_pandas()

        if isinstance(data, (DataFrame, pd.DataFrame)):
            if isinstance(data, pd.DataFrame):
                data = self.from_pandas(data, nan_as_null=nan_as_null)

            if index is not None:
                if not data.index.equals(index):
                    data = data.reindex(index)
                    index = data._index
                else:
                    index = as_index(index)
            else:
                index = data._index

            self._index = index

            if columns is not None:
                self._data = data._data
                self._reindex(
                    columns=columns, index=index, deep=False, inplace=True
                )
            else:
                self._data = data._data
                self.columns = data.columns
        elif isinstance(data, (cudf.Series, pd.Series)):
            if isinstance(data, pd.Series):
                data = cudf.Series.from_pandas(data, nan_as_null=nan_as_null)

            name = data.name or 0
            self._init_from_dict_like(
                {name: data},
                index=index,
                columns=columns,
                nan_as_null=nan_as_null,
            )
        elif data is None:
            if index is None:
                self._index = RangeIndex(0)
            else:
                self._index = as_index(index)
            if columns is not None:
                self._data = ColumnAccessor(
                    {
                        k: column.column_empty(
                            len(self), dtype="object", masked=True
                        )
                        for k in columns
                    }
                )
        elif hasattr(data, "__cuda_array_interface__"):
            arr_interface = data.__cuda_array_interface__

            # descr is an optional field of the _cuda_ary_iface_
            if "descr" in arr_interface:
                if len(arr_interface["descr"]) == 1:
                    new_df = self._from_arrays(
                        data, index=index, columns=columns
                    )
                else:
                    new_df = self.from_records(
                        data, index=index, columns=columns
                    )
            else:
                new_df = self._from_arrays(data, index=index, columns=columns)

            self._data = new_df._data
            self.index = new_df._index
            self.columns = new_df.columns
        elif hasattr(data, "__array_interface__"):
            arr_interface = data.__array_interface__
            if len(arr_interface["descr"]) == 1:
                # not record arrays
                new_df = self._from_arrays(data, index=index, columns=columns)
            else:
                new_df = self.from_records(data, index=index, columns=columns)
            self._data = new_df._data
            self.index = new_df._index
            self.columns = new_df.columns
        else:
            if is_list_like(data):
                if len(data) > 0 and is_scalar(data[0]):
                    if columns is not None:
                        data = dict(zip(columns, [data]))
                    else:
                        data = dict(enumerate([data]))
                    new_df = DataFrame(data=data, index=index)

                    self._data = new_df._data
                    self.index = new_df._index
                    self.columns = new_df.columns
                elif len(data) > 0 and isinstance(data[0], Series):
                    self._init_from_series_list(
                        data=data, columns=columns, index=index
                    )
                else:
                    self._init_from_list_like(
                        data, index=index, columns=columns
                    )

            else:
                if not is_dict_like(data):
                    raise TypeError("data must be list or dict-like")

                self._init_from_dict_like(
                    data, index=index, columns=columns, nan_as_null=nan_as_null
                )

        if dtype:
            self._data = self.astype(dtype)._data

    def _init_from_series_list(self, data, columns, index):
        if index is None:
            # When `index` is `None`, the final index of
            # resulting dataframe will be union of
            # all Series's names.
            final_index = as_index(_get_union_of_series_names(data))
        else:
            # When an `index` is passed, the final index of
            # resulting dataframe will be whatever
            # index passed, but will need
            # shape validations - explained below
            data_length = len(data)
            index_length = len(index)
            if data_length != index_length:
                # If the passed `index` length doesn't match
                # length of Series objects in `data`, we must
                # check if `data` can be duplicated/expanded
                # to match the length of index. For that we
                # check if the length of index is a factor
                # of length of data.
                #
                # 1. If yes, we extend data
                # until length of data is equal to length of index.
                # 2. If no, we throw an error stating the
                # shape of resulting `data` and `index`

                # Simple example
                # >>> import pandas as pd
                # >>> s = pd.Series([1, 2, 3])
                # >>> pd.DataFrame([s], index=['a', 'b'])
                #    0  1  2
                # a  1  2  3
                # b  1  2  3
                # >>> pd.DataFrame([s], index=['a', 'b', 'c'])
                #    0  1  2
                # a  1  2  3
                # b  1  2  3
                # c  1  2  3
                if index_length % data_length == 0:
                    initial_data = data
                    data = []
                    for _ in range(int(index_length / data_length)):
                        data.extend([o for o in initial_data])
                else:
                    raise ValueError(
                        f"Shape of passed values is "
                        f"{(data_length, len(data[0]))}, "
                        f"indices imply {(index_length, len(data[0]))}"
                    )

            final_index = as_index(index)

        series_lengths = list(map(lambda x: len(x), data))
        data = numeric_normalize_types(*data)
        if series_lengths.count(series_lengths[0]) == len(series_lengths):
            # Calculating the final dataframe columns by
            # getting union of all `index` of the Series objects.
            final_columns = _get_union_of_indices([d.index for d in data])

            for idx, series in enumerate(data):
                if not series.index.is_unique:
                    raise ValueError(
                        "Reindexing only valid with uniquely valued Index "
                        "objects"
                    )
                if not series.index.equals(final_columns):
                    series = series.reindex(final_columns)
                self._data[idx] = column.as_column(series._column)

            # Setting `final_columns` to self._index so
            # that the resulting `transpose` will be have
            # columns set to `final_columns`
            self._index = as_index(final_columns)

            transpose = self.T
        else:
            concat_df = cudf.concat(data, axis=1)

            if concat_df.columns.dtype == "object":
                concat_df.columns = concat_df.columns.astype("str")

            transpose = concat_df.T

        transpose._index = final_index
        self._data = transpose._data
        self._index = transpose._index

        # If `columns` is passed, the result dataframe
        # contain a dataframe with only the
        # specified `columns` in the same order.
        if columns is not None:
            for col_name in columns:
                if col_name not in self._data:
                    self._data[col_name] = column.column_empty(
                        row_count=len(self), dtype=None, masked=True
                    )
            self._data = self._data.select_by_label(columns)

    def _init_from_list_like(self, data, index=None, columns=None):
        if index is None:
            index = RangeIndex(start=0, stop=len(data))
        else:
            index = as_index(index)

        self._index = as_index(index)
        # list-of-dicts case
        if len(data) > 0 and isinstance(data[0], dict):
            data = DataFrame.from_pandas(pd.DataFrame(data))
            self._data = data._data
        # interval in a list
        elif len(data) > 0 and isinstance(data[0], pd._libs.interval.Interval):
            data = DataFrame.from_pandas(pd.DataFrame(data))
            self._data = data._data
        else:
            data = list(itertools.zip_longest(*data))

            if columns is not None and len(data) == 0:
                data = [
                    cudf.core.column.column_empty(row_count=0, dtype=None)
                    for _ in columns
                ]

            for col_name, col in enumerate(data):
                self._data[col_name] = column.as_column(col)

        if columns is not None:
            if len(columns) != len(data):
                raise ValueError(
                    f"Shape of passed values is ({len(index)}, {len(data)}), "
                    f"indices imply ({len(index)}, {len(columns)})."
                )

            self.columns = columns

    def _init_from_dict_like(
        self, data, index=None, columns=None, nan_as_null=None
    ):
        if columns is not None:
            # remove all entries in `data` that are
            # not in `columns`
            keys = [key for key in data.keys() if key in columns]
            extra_cols = [col for col in columns if col not in keys]
            if keys:
                # if keys is non-empty,
                # add null columns for all values
                # in `columns` that don't exist in `keys`:
                data = {key: data[key] for key in keys}
                data.update({key: None for key in extra_cols})
            else:
                # If keys is empty, none of the data keys match the columns, so
                # we need to create an empty DataFrame. To match pandas, the
                # size of the dataframe must match the provided index, so we
                # need to return a masked array of nulls if an index is given.
                row_count = 0 if index is None else len(index)
                masked = index is not None
                data = {
                    key: cudf.core.column.column_empty(
                        row_count=row_count, dtype=None, masked=masked,
                    )
                    for key in extra_cols
                }

        data, index = self._align_input_series_indices(data, index=index)

        if index is None:
            num_rows = 0
            if data:
                col_name = next(iter(data))
                if is_scalar(data[col_name]):
                    num_rows = num_rows or 1
                else:
                    data[col_name] = column.as_column(
                        data[col_name], nan_as_null=nan_as_null
                    )
                    num_rows = len(data[col_name])
            self._index = RangeIndex(0, num_rows)
        else:
            self._index = as_index(index)

        if len(data):
            self._data.multiindex = True
            for (i, col_name) in enumerate(data):
                self._data.multiindex = self._data.multiindex and isinstance(
                    col_name, tuple
                )
                self.insert(
                    i, col_name, data[col_name], nan_as_null=nan_as_null
                )

        if columns is not None:
            self.columns = columns

    @classmethod
    def _from_data(
        cls,
        data: MutableMapping,
        index: Optional[BaseIndex] = None,
        columns: Any = None,
    ) -> DataFrame:
        out = super()._from_data(data, index)
        if index is None:
            out.index = RangeIndex(out._data.nrows)
        if columns is not None:
            out.columns = columns
        return out

    @staticmethod
    def _align_input_series_indices(data, index):
        data = data.copy()

        input_series = [
            Series(val)
            for val in data.values()
            if isinstance(val, (pd.Series, Series))
        ]

        if input_series:
            if index is not None:
                aligned_input_series = [
                    sr._align_to_index(index, how="right", sort=False)
                    for sr in input_series
                ]

            else:
                aligned_input_series = cudf.core.series._align_indices(
                    input_series
                )
                index = aligned_input_series[0].index

            for name, val in data.items():
                if isinstance(val, (pd.Series, Series)):
                    data[name] = aligned_input_series.pop(0)

        return data, index

    # The `constructor*` properties are used by `dask` (and `dask_cudf`)
    @property
    def _constructor(self):
        return DataFrame

    @property
    def _constructor_sliced(self):
        return Series

    @property
    def _constructor_expanddim(self):
        raise NotImplementedError(
            "_constructor_expanddim not supported for DataFrames!"
        )

    def serialize(self):
        header, frames = super().serialize()

        header["index"], index_frames = self._index.serialize()
        header["index_frame_count"] = len(index_frames)
        # For backwards compatibility with older versions of cuDF, index
        # columns are placed before data columns.
        frames = index_frames + frames

        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        index_nframes = header["index_frame_count"]
        obj = super().deserialize(
            header, frames[header["index_frame_count"] :]
        )

        idx_typ = pickle.loads(header["index"]["type-serialized"])
        index = idx_typ.deserialize(header["index"], frames[:index_nframes])
        obj._index = index

        return obj

    @property
    def dtypes(self):
        """
        Return the dtypes in this object.

        Returns
        -------
        pandas.Series
            The data type of each column.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> df = cudf.DataFrame({'float': [1.0],
        ...                    'int': [1],
        ...                    'datetime': [pd.Timestamp('20180310')],
        ...                    'string': ['foo']})
        >>> df
           float  int   datetime string
        0    1.0    1 2018-03-10    foo
        >>> df.dtypes
        float              float64
        int                  int64
        datetime    datetime64[us]
        string              object
        dtype: object
        """
        return pd.Series(self._dtypes)

    @property
    def _dtypes(self):
        return dict(
            zip(self._data.names, (col.dtype for col in self._data.columns))
        )

    @property
    def ndim(self):
        """Dimension of the data. DataFrame ndim is always 2."""
        return 2

    def __dir__(self):
        # Add the columns of the DataFrame to the dir output.
        o = set(dir(type(self)))
        o.update(self.__dict__)
        o.update(
            c for c in self.columns if isinstance(c, str) and c.isidentifier()
        )
        return list(o)

    def __setattr__(self, key, col):
        try:
            # Preexisting attributes may be set. We cannot rely on checking the
            # `_PROTECTED_KEYS` because we must also allow for settable
            # properties, and we must call object.__getattribute__ to bypass
            # the `__getitem__` behavior inherited from `GetAttrGetItemMixin`.
            object.__getattribute__(self, key)
            super().__setattr__(key, col)
        except AttributeError:
            if key not in self._PROTECTED_KEYS:
                try:
                    # Check key existence.
                    self[key]
                    # If a column already exists, set it.
                    self[key] = col
                    return
                except KeyError:
                    pass

            # Set a new attribute that is not already a column.
            super().__setattr__(key, col)

    @annotate("DATAFRAME_GETITEM", color="blue", domain="cudf_python")
    def __getitem__(self, arg):
        """
        If *arg* is a ``str`` or ``int`` type, return the column Series.
        If *arg* is a ``slice``, return a new DataFrame with all columns
        sliced to the specified range.
        If *arg* is an ``array`` containing column names, return a new
        DataFrame with the corresponding columns.
        If *arg* is a ``dtype.bool array``, return the rows marked True

        Examples
        --------
        >>> df = DataFrame([('a', list(range(20))),
        ...                 ('b', list(range(20))),
        ...                 ('c', list(range(20)))])
        >>> df[:4]    # get first 4 rows of all columns
           a  b  c
        0  0  0  0
        1  1  1  1
        2  2  2  2
        3  3  3  3
        >>> df[-5:]  # get last 5 rows of all columns
             a   b   c
        15  15  15  15
        16  16  16  16
        17  17  17  17
        18  18  18  18
        19  19  19  19
        >>> df[['a', 'c']] # get columns a and c
           a  c
        0  0  0
        1  1  1
        2  2  2
        3  3  3
        4  4  4
        5  5  5
        6  6  6
        7  7  7
        8  8  8
        9  9  9
        >>> df[[True, False, True, False]] # mask the entire dataframe,
        # returning the rows specified in the boolean mask
        """
        if _is_scalar_or_zero_d_array(arg) or isinstance(arg, tuple):
            return self._get_columns_by_label(arg, downcast=True)

        elif isinstance(arg, slice):
            return self._slice(arg)

        elif can_convert_to_column(arg):
            mask = arg
            if is_list_like(mask):
                mask = cudf.utils.utils._create_pandas_series(data=mask)
            if mask.dtype == "bool":
                return self._apply_boolean_mask(mask)
            else:
                return self._get_columns_by_label(mask)
        elif isinstance(arg, DataFrame):
            return self.where(arg)
        else:
            raise TypeError(
                f"__getitem__ on type {type(arg)} is not supported"
            )

    @annotate("DATAFRAME_SETITEM", color="blue", domain="cudf_python")
    def __setitem__(self, arg, value):
        """Add/set column by *arg or DataFrame*"""
        if isinstance(arg, DataFrame):
            # not handling set_item where arg = df & value = df
            if isinstance(value, DataFrame):
                raise TypeError(
                    f"__setitem__ with arg = {type(value)} and "
                    f"value = {type(arg)} is not supported"
                )
            else:
                for col_name in self._data:
                    scatter_map = arg[col_name]
                    if is_scalar(value):
                        self._data[col_name][scatter_map] = value
                    else:

                        self._data[col_name][scatter_map] = column.as_column(
                            value
                        )[scatter_map]
        elif is_scalar(arg) or isinstance(arg, tuple):
            if isinstance(value, DataFrame):
                _setitem_with_dataframe(
                    input_df=self,
                    replace_df=value,
                    input_cols=[arg],
                    mask=None,
                )
            else:
                if arg in self._data:
                    if len(self) == 0:
                        if isinstance(value, (pd.Series, Series)):
                            self._index = as_index(value.index)
                        elif len(value) > 0:
                            self._index = RangeIndex(start=0, stop=len(value))
                        value = column.as_column(value)
                        new_data = self._data.__class__()
                        for key in self._data:
                            if key == arg:
                                new_data[key] = value
                            else:
                                new_data[key] = column.column_empty_like(
                                    self._data[key],
                                    masked=True,
                                    newsize=len(value),
                                )

                        self._data = new_data
                        return
                    elif isinstance(value, (pd.Series, Series)):
                        value = Series(value)._align_to_index(
                            self._index,
                            how="right",
                            sort=False,
                            allow_non_unique=True,
                        )
                    if is_scalar(value):
                        self._data[arg][:] = value
                    else:
                        value = as_column(value)
                        self._data[arg] = value
                else:
                    # disc. with pandas here
                    # pandas raises key error here
                    self.insert(len(self._data), arg, value)

        elif can_convert_to_column(arg):
            mask = arg
            if is_list_like(mask):
                mask = np.array(mask)

            if mask.dtype == "bool":
                mask = column.as_column(arg)

                if isinstance(value, DataFrame):
                    _setitem_with_dataframe(
                        input_df=self,
                        replace_df=value,
                        input_cols=None,
                        mask=mask,
                    )
                else:
                    if not is_scalar(value):
                        value = column.as_column(value)[mask]
                    for col_name in self._data:
                        self._data[col_name][mask] = value
            else:
                if isinstance(value, DataFrame):
                    _setitem_with_dataframe(
                        input_df=self,
                        replace_df=value,
                        input_cols=arg,
                        mask=None,
                    )
                else:
                    for col in arg:
                        if is_scalar(value):
                            self._data[col] = column.full(
                                size=len(self), fill_value=value
                            )
                        else:
                            self._data[col] = column.as_column(value)

        else:
            raise TypeError(
                f"__setitem__ on type {type(arg)} is not supported"
            )

    def __delitem__(self, name):
        """
        Drop the given column by *name*.
        """
        self._drop_column(name)

    def _slice(self: T, arg: slice) -> T:
        """
        _slice : slice the frame as per the arg

        Parameters
        ----------
        arg : should always be of type slice

        """
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

        # For decreasing slices, terminal at before-the-zero
        # position is preserved.
        if start < 0:
            start = start + num_rows
        if stop < 0 and not (stride < 0 and stop == -1):
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
                result = self._from_data(
                    *libcudf.copying.table_slice(
                        self, [start, stop], keep_index
                    )[0]
                )

                result._copy_type_metadata(self, include_index=keep_index)
                if self.index is not None:
                    if keep_index:
                        result._index.names = self.index.names
                    else:
                        # Adding index of type RangeIndex back to
                        # result
                        result.index = self.index[start:stop]
                result.columns = self.columns
                return result

    def memory_usage(self, index=True, deep=False):
        """
        Return the memory usage of each column in bytes.
        The memory usage can optionally include the contribution of
        the index and elements of `object` dtype.

        Parameters
        ----------
        index : bool, default True
            Specifies whether to include the memory usage of the DataFrame's
            index in returned Series. If ``index=True``, the memory usage of
            the index is the first item in the output.
        deep : bool, default False
            If True, introspect the data deeply by interrogating
            `object` dtypes for system-level memory consumption, and include
            it in the returned values.

        Returns
        -------
        Series
            A Series whose index is the original column names and whose values
            is the memory usage of each column in bytes.

        Examples
        --------
        >>> dtypes = ['int64', 'float64', 'object', 'bool']
        >>> data = dict([(t, np.ones(shape=5000).astype(t))
        ...              for t in dtypes])
        >>> df = cudf.DataFrame(data)
        >>> df.head()
           int64  float64  object  bool
        0      1      1.0     1.0  True
        1      1      1.0     1.0  True
        2      1      1.0     1.0  True
        3      1      1.0     1.0  True
        4      1      1.0     1.0  True
        >>> df.memory_usage(index=False)
        int64      40000
        float64    40000
        object     40000
        bool        5000
        dtype: int64
        Use a Categorical for efficient storage of an object-dtype column with
        many repeated values.
        >>> df['object'].astype('category').memory_usage(deep=True)
        5048
        """
        if deep:
            warnings.warn(
                "The deep parameter is ignored and is only included "
                "for pandas compatibility."
            )
        ind = list(self.columns)
        sizes = [col.memory_usage() for col in self._data.columns]
        if index:
            ind.append("Index")
            ind = cudf.Index(ind, dtype="str")
            sizes.append(self.index.memory_usage())
        return Series(sizes, index=ind)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__" and hasattr(cudf, ufunc.__name__):
            func = getattr(cudf, ufunc.__name__)
            return func(self)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):

        cudf_df_module = DataFrame
        cudf_series_module = Series

        for submodule in func.__module__.split(".")[1:]:
            # point cudf to the correct submodule
            if hasattr(cudf_df_module, submodule):
                cudf_df_module = getattr(cudf_df_module, submodule)
            else:
                return NotImplemented

        fname = func.__name__

        handled_types = [cudf_df_module, cudf_series_module]

        for t in types:
            if t not in handled_types:
                return NotImplemented

        if hasattr(cudf_df_module, fname):
            cudf_func = getattr(cudf_df_module, fname)
            # Handle case if cudf_func is same as numpy function
            if cudf_func is func:
                return NotImplemented
            # numpy returns an array from the dot product of two dataframes
            elif (
                func is np.dot
                and isinstance(args[0], (DataFrame, pd.DataFrame))
                and isinstance(args[1], (DataFrame, pd.DataFrame))
            ):
                return cudf_func(*args, **kwargs).values
            else:
                return cudf_func(*args, **kwargs)
        else:
            return NotImplemented

    # The _get_numeric_data method is necessary for dask compatibility.
    def _get_numeric_data(self):
        """Return a dataframe with only numeric data types"""
        columns = [
            c
            for c, dt in self.dtypes.items()
            if dt != object and not is_categorical_dtype(dt)
        ]
        return self[columns]

    def assign(self, **kwargs):
        """
        Assign columns to DataFrame from keyword arguments.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df = df.assign(a=[0, 1, 2], b=[3, 4, 5])
        >>> df
           a  b
        0  0  3
        1  1  4
        2  2  5
        """
        new = self.copy()
        for k, v in kwargs.items():
            new[k] = v
        return new

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

        # flag to indicate if all DataFrame's have
        # RangeIndex as their index
        are_all_range_index = False

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

            are_all_range_index = (
                True if i == 0 else are_all_range_index
            ) and isinstance(obj.index, cudf.RangeIndex)

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
                if are_all_range_index
                or (ignore_index and not empty_has_index)
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
                table_index = Frame(
                    data=dict(
                        zip(
                            indices[:first_data_column_position],
                            cols[:first_data_column_position],
                        )
                    )
                )
            tables.append(
                Frame(
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
        out = cls._from_data(
            *libcudf.concat.concat_tables(
                tables, ignore_index=ignore_index or are_all_range_index
            )
        )

        # If ignore_index is True, all input frames are empty, and at
        # least one input frame has an index, assign a new RangeIndex
        # to the result frame.
        if empty_has_index and num_empty_input_frames == len(objs):
            out._index = cudf.RangeIndex(result_index_length)
        elif are_all_range_index and not ignore_index:
            out._index = cudf.core.index.GenericIndex._concat(
                [o._index for o in objs]
            )

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
            if not isinstance(out._index, MultiIndex) and is_categorical_dtype(
                out._index._values.dtype
            ):
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

    def astype(self, dtype, copy=False, errors="raise", **kwargs):
        """
        Cast the DataFrame to the given dtype

        Parameters
        ----------

        dtype : data type, or dict of column name -> data type
            Use a numpy.dtype or Python type to cast entire DataFrame object to
            the same type. Alternatively, use ``{col: dtype, ...}``, where col
            is a column label and dtype is a numpy.dtype or Python type
            to cast one or more of the DataFrame's columns to
            column-specific types.
        copy : bool, default False
            Return a deep-copy when ``copy=True``. Note by default
            ``copy=False`` setting is used and hence changes to
            values then may propagate to other cudf objects.
        errors : {'raise', 'ignore', 'warn'}, default 'raise'
            Control raising of exceptions on invalid data for provided dtype.

            -   ``raise`` : allow exceptions to be raised
            -   ``ignore`` : suppress exceptions. On error return original
                object.
            -   ``warn`` : prints last exceptions as warnings and
                return original object.
        **kwargs : extra arguments to pass on to the constructor

        Returns
        -------
        casted : DataFrame

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [10, 20, 30], 'b': [1, 2, 3]})
        >>> df
            a  b
        0  10  1
        1  20  2
        2  30  3
        >>> df.dtypes
        a    int64
        b    int64
        dtype: object

        Cast all columns to `int32`:

        >>> df.astype('int32').dtypes
        a    int32
        b    int32
        dtype: object

        Cast `a` to `float32` using a dictionary:

        >>> df.astype({'a': 'float32'}).dtypes
        a    float32
        b      int64
        dtype: object
        >>> df.astype({'a': 'float32'})
              a  b
        0  10.0  1
        1  20.0  2
        2  30.0  3
        """
        result = DataFrame(index=self.index)

        if is_dict_like(dtype):
            current_cols = self._data.names
            if len(set(dtype.keys()) - set(current_cols)) > 0:
                raise KeyError(
                    "Only a column name can be used for the "
                    "key in a dtype mappings argument."
                )
            for col_name in current_cols:
                if col_name in dtype:
                    result._data[col_name] = self._data[col_name].astype(
                        dtype=dtype[col_name],
                        errors=errors,
                        copy=copy,
                        **kwargs,
                    )
                else:
                    result._data[col_name] = (
                        self._data[col_name].copy(deep=True)
                        if copy
                        else self._data[col_name]
                    )
        else:
            for col in self._data:
                result._data[col] = self._data[col].astype(
                    dtype=dtype, **kwargs
                )

        return result

    def _clean_renderable_dataframe(self, output):
        """
        This method takes in partial/preprocessed dataframe
        and returns correct representation of it with correct
        dimensions (rows x columns)
        """

        max_rows = get_option("display.max_rows")
        min_rows = get_option("display.min_rows")
        max_cols = get_option("display.max_columns")
        max_colwidth = get_option("display.max_colwidth")
        show_dimensions = get_option("display.show_dimensions")
        if get_option("display.expand_frame_repr"):
            width, _ = console.get_console_size()
        else:
            width = None

        output = output.to_pandas().to_string(
            max_rows=max_rows,
            min_rows=min_rows,
            max_cols=max_cols,
            line_width=width,
            max_colwidth=max_colwidth,
            show_dimensions=show_dimensions,
        )

        lines = output.split("\n")

        if lines[-1].startswith("["):
            lines = lines[:-1]
            lines.append(
                "[%d rows x %d columns]" % (len(self), len(self._data.names))
            )
        return "\n".join(lines)

    def _clean_nulls_from_dataframe(self, df):
        """
        This function converts all ``null`` values to ``<NA>`` for
        representation as a string in `__repr__`.

        Since we utilize Pandas `__repr__` at all places in our code
        for formatting purposes, we convert columns to `str` dtype for
        filling with `<NA>` values.
        """
        for col in df._data:
            if is_list_dtype(df._data[col]) or is_struct_dtype(df._data[col]):
                # TODO we need to handle this
                pass
            elif df._data[col].has_nulls():
                df[col] = df._data[col].astype("str").fillna(cudf._NA_REP)
            else:
                df[col] = df._data[col]

        return df

    def _get_renderable_dataframe(self):
        """
        takes rows and columns from pandas settings or estimation from size.
        pulls quadrents based off of some known parameters then style for
        multiindex as well producing an efficient representative string
        for printing with the dataframe.
        """
        max_rows = pd.options.display.max_rows
        nrows = np.max([len(self) if max_rows is None else max_rows, 1])
        if pd.options.display.max_rows == 0:
            nrows = len(self)
        ncols = (
            pd.options.display.max_columns
            if pd.options.display.max_columns
            else pd.options.display.width / 2
        )

        if len(self) <= nrows and len(self._data.names) <= ncols:
            output = self.copy(deep=False)
        elif self.empty and len(self.index) > 0:
            max_seq_items = pd.options.display.max_seq_items
            # Incase of Empty DataFrame with index, Pandas prints
            # first `pd.options.display.max_seq_items` index values
            # followed by ... To obtain ... at the end of index list,
            # adding 1 extra value.
            # If `pd.options.display.max_seq_items` is None,
            # entire sequence/Index is to be printed.
            # Note : Pandas truncates the dimensions at the end of
            # the resulting dataframe when `display.show_dimensions`
            # is set to truncate. Hence to display the dimensions we
            # need to extract maximum of `max_seq_items` and `nrows`
            # and have 1 extra value for ... to show up in the output
            # string.
            if max_seq_items is not None:
                output = self.head(max(max_seq_items, nrows) + 1)
            else:
                output = self.copy(deep=False)
        else:
            left_cols = len(self._data.names)
            right_cols = 0
            upper_rows = len(self)
            lower_rows = 0
            if len(self) > nrows and nrows > 0:
                upper_rows = int(nrows / 2.0) + 1
                lower_rows = upper_rows + (nrows % 2)
            if len(self._data.names) > ncols:
                right_cols = len(self._data.names) - int(ncols / 2.0)
                # adjust right columns for output if multiindex.
                right_cols = (
                    right_cols - 1
                    if isinstance(self.index, MultiIndex)
                    else right_cols
                )
                left_cols = int(ncols / 2.0) + 1
            if right_cols > 0:
                # Pick ncols - left_cols number of columns
                # from the right side/from the end.
                right_cols = -(int(ncols) - left_cols + 1)
            else:
                # If right_cols is 0 or negative, it means
                # self has lesser number of columns than ncols.
                # Hence assign len(self._data.names) which
                # will result in empty `*_right` quadrants.
                # This is because `*_left` quadrants will
                # contain all columns.
                right_cols = len(self._data.names)

            upper_left = self.head(upper_rows).iloc[:, :left_cols]
            upper_right = self.head(upper_rows).iloc[:, right_cols:]
            lower_left = self.tail(lower_rows).iloc[:, :left_cols]
            lower_right = self.tail(lower_rows).iloc[:, right_cols:]

            upper = cudf.concat([upper_left, upper_right], axis=1)
            lower = cudf.concat([lower_left, lower_right], axis=1)
            output = cudf.concat([upper, lower])

        output = self._clean_nulls_from_dataframe(output)
        output._index = output._index._clean_nulls_from_index()

        return output

    def __repr__(self):
        output = self._get_renderable_dataframe()
        return self._clean_renderable_dataframe(output)

    def _repr_html_(self):
        lines = (
            self._get_renderable_dataframe()
            .to_pandas()
            ._repr_html_()
            .split("\n")
        )
        if lines[-2].startswith("<p>"):
            lines = lines[:-2]
            lines.append(
                "<p>%d rows × %d columns</p>"
                % (len(self), len(self._data.names))
            )
            lines.append("</div>")
        return "\n".join(lines)

    def _repr_latex_(self):
        return self._get_renderable_dataframe().to_pandas()._repr_latex_()

    def _get_columns_by_label(self, labels, downcast=False):
        """
        Return columns of dataframe by `labels`

        If downcast is True, try and downcast from a DataFrame to a Series
        """
        new_data = super()._get_columns_by_label(labels, downcast)
        if downcast:
            if is_scalar(labels):
                nlevels = 1
            elif isinstance(labels, tuple):
                nlevels = len(labels)
            if self._data.multiindex is False or nlevels == self._data.nlevels:
                out = self._constructor_sliced._from_data(
                    new_data, index=self.index, name=labels
                )
                return out
        out = self.__class__._from_data(
            new_data, index=self.index, columns=new_data.to_pandas_index()
        )
        return out

    def _binaryop(
        self,
        other: Any,
        fn: str,
        fill_value: Any = None,
        reflect: bool = False,
        can_reindex: bool = False,
        *args,
        **kwargs,
    ):
        lhs, rhs = self, other

        if _is_scalar_or_zero_d_array(rhs):
            rhs = [rhs] * lhs._num_columns

        # For columns that exist in rhs but not lhs, we swap the order so that
        # we can always assume that left has a binary operator. This
        # implementation assumes that binary operations between a column and
        # NULL are always commutative, even for binops (like subtraction) that
        # are normally anticommutative.
        if isinstance(rhs, Sequence):
            # TODO: Consider validating sequence length (pandas does).
            operands = {
                name: (left, right, reflect, fill_value)
                for right, (name, left) in zip(rhs, lhs._data.items())
            }
        elif isinstance(rhs, DataFrame):
            if (
                not can_reindex
                and fn in cudf.utils.utils._EQUALITY_OPS
                and (
                    not lhs.columns.equals(rhs.columns)
                    or not lhs.index.equals(rhs.index)
                )
            ):
                raise ValueError(
                    "Can only compare identically-labeled " "DataFrame objects"
                )

            lhs, rhs = _align_indices(lhs, rhs)

            operands = {
                name: (
                    lcol,
                    rhs._data[name]
                    if name in rhs._data
                    else (fill_value or None),
                    reflect,
                    fill_value if name in rhs._data else None,
                )
                for name, lcol in lhs._data.items()
            }
            for name, col in rhs._data.items():
                if name not in lhs._data:
                    operands[name] = (
                        col,
                        (fill_value or None),
                        not reflect,
                        None,
                    )
        elif isinstance(rhs, Series):
            # Note: This logic will need updating if any of the user-facing
            # binop methods (e.g. DataFrame.add) ever support axis=0/rows.
            right_dict = dict(zip(rhs.index.values_host, rhs.values_host))
            left_cols = lhs._column_names
            # mypy thinks lhs._column_names is a List rather than a Tuple, so
            # we have to ignore the type check.
            result_cols = left_cols + tuple(  # type: ignore
                col for col in right_dict if col not in left_cols
            )
            operands = {}
            for col in result_cols:
                if col in left_cols:
                    left = lhs._data[col]
                    right = right_dict[col] if col in right_dict else None
                else:
                    # We match pandas semantics here by performing binops
                    # between a NaN (not NULL!) column and the actual values,
                    # which results in nans, the pandas output.
                    left = as_column(np.nan, length=lhs._num_rows)
                    right = right_dict[col]
                operands[col] = (left, right, reflect, fill_value)
        else:
            return NotImplemented

        return self._from_data(
            ColumnAccessor(type(self)._colwise_binop(operands, fn)),
            index=lhs._index,
        )

    def update(
        self,
        other,
        join="left",
        overwrite=True,
        filter_func=None,
        errors="ignore",
    ):
        """
        Modify a DataFrame in place using non-NA values from another DataFrame.

        Aligns on indices. There is no return value.

        Parameters
        ----------
        other : DataFrame, or object coercible into a DataFrame
            Should have at least one matching index/column label with the
            original DataFrame. If a Series is passed, its name attribute must
            be set, and that will be used as the column name to align with the
            original DataFrame.

        join : {'left'}, default 'left'
            Only left join is implemented, keeping the index and
            columns of the original object.

        overwrite : {True, False}, default True
            How to handle non-NA values for overlapping keys:
            True: overwrite original DataFrame's values with values from other.
            False: only update values that are NA in the original DataFrame.

        filter_func : None
            filter_func is not supported yet
            Return True for values that should be updated.S

        errors : {'raise', 'ignore'}, default 'ignore'
            If 'raise', will raise a ValueError if the DataFrame and other
            both contain non-NA data in the same place.


        Returns
        -------
        None : method directly changes calling object

        Raises
        -------
        ValueError
            - When ``errors`` = 'raise' and there's overlapping non-NA data.
            - When ``errors`` is not either 'ignore' or 'raise'

        NotImplementedError
            - If ``join`` != 'left'
        """
        # TODO: Support other joins
        if join != "left":
            raise NotImplementedError("Only left join is supported")
        if errors not in {"ignore", "raise"}:
            raise ValueError(
                "The parameter errors must be either 'ignore' or 'raise'"
            )
        if filter_func is not None:
            raise NotImplementedError("filter_func is not supported yet")

        if not isinstance(other, DataFrame):
            other = DataFrame(other)

        if not self.columns.equals(other.columns):
            other = other.reindex(self.columns, axis=1)
        if not self.index.equals(other.index):
            other = other.reindex(self.index, axis=0)

        source_df = self.copy(deep=False)
        for col in source_df._column_names:
            this = source_df[col]
            that = other[col]

            if errors == "raise":
                mask_this = that.notna()
                mask_that = this.notna()
                if (mask_this & mask_that).any():
                    raise ValueError("Data overlaps.")

            if overwrite:
                mask = that.isna()
            else:
                mask = this.notna()

            # don't overwrite columns unnecessarily
            if mask.all():
                continue
            source_df[col] = source_df[col].where(mask, that)

        self._mimic_inplace(source_df, inplace=True)

    def __iter__(self):
        return iter(self.columns)

    def iteritems(self):
        """Iterate over column names and series pairs"""
        for k in self:
            yield (k, self[k])

    def equals(self, other, **kwargs):
        ret = super().equals(other)
        # If all other checks matched, validate names.
        if ret:
            for self_name, other_name in zip(
                self._data.names, other._data.names
            ):
                if self_name != other_name:
                    ret = False
                    break
        return ret

    @property
    def iat(self):
        """
        Alias for ``DataFrame.iloc``; provided for compatibility with Pandas.
        """
        return self.iloc

    @property
    def at(self):
        """
        Alias for ``DataFrame.loc``; provided for compatibility with Pandas.
        """
        return self.loc

    @property  # type: ignore
    @annotate("DATAFRAME_COLUMNS_GETTER", color="yellow", domain="cudf_python")
    def columns(self):
        """Returns a tuple of columns"""
        return self._data.to_pandas_index()

    @columns.setter  # type: ignore
    @annotate("DATAFRAME_COLUMNS_SETTER", color="yellow", domain="cudf_python")
    def columns(self, columns):
        if isinstance(columns, cudf.BaseIndex):
            columns = columns.to_pandas()
        if columns is None:
            columns = pd.Index(range(len(self._data.columns)))
        is_multiindex = isinstance(columns, pd.MultiIndex)

        if isinstance(
            columns, (Series, cudf.Index, cudf.core.column.ColumnBase)
        ):
            columns = pd.Index(columns.to_numpy(), tupleize_cols=is_multiindex)
        elif not isinstance(columns, pd.Index):
            columns = pd.Index(columns, tupleize_cols=is_multiindex)

        if not len(columns) == len(self._data.names):
            raise ValueError(
                f"Length mismatch: expected {len(self._data.names)} elements, "
                f"got {len(columns)} elements"
            )

        data = dict(zip(columns, self._data.columns))
        if len(columns) != len(data):
            raise ValueError("Duplicate column names are not allowed")

        self._data = ColumnAccessor(
            data, multiindex=is_multiindex, level_names=columns.names,
        )

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

            idx_dtype_match = (df.index.nlevels == index.nlevels) and all(
                left_dtype == right_dtype
                for left_dtype, right_dtype in zip(
                    (col.dtype for col in df.index._data.columns),
                    (col.dtype for col in index._data.columns),
                )
            )

            if not idx_dtype_match:
                columns = (
                    columns if columns is not None else list(df._column_names)
                )
                df = DataFrame()
            else:
                df = DataFrame(None, index).join(df, how="left", sort=True)
                # double-argsort to map back from sorted to unsorted positions
                df = df.take(index.argsort(ascending=True).argsort())

        index = index if index is not None else df.index
        names = columns if columns is not None else list(df._data.names)
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

        result = self.__class__._from_data(
            data=cudf.core.column_accessor.ColumnAccessor(
                cols,
                multiindex=self._data.multiindex,
                level_names=self._data.level_names,
            ),
            index=index,
        )

        return self._mimic_inplace(result, inplace=inplace)

    def reindex(
        self, labels=None, axis=None, index=None, columns=None, copy=True
    ):
        """
        Return a new DataFrame whose axes conform to a new index

        ``DataFrame.reindex`` supports two calling conventions:
            - ``(index=index_labels, columns=column_names)``
            - ``(labels, axis={0 or 'index', 1 or 'columns'})``

        Parameters
        ----------
        labels : Index, Series-convertible, optional, default None
        axis : {0 or 'index', 1 or 'columns'}, optional, default 0
        index : Index, Series-convertible, optional, default None
            Shorthand for ``df.reindex(labels=index_labels, axis=0)``
        columns : array-like, optional, default None
            Shorthand for ``df.reindex(labels=column_names, axis=1)``
        copy : boolean, optional, default True

        Returns
        -------
        A DataFrame whose axes conform to the new index(es)

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df['key'] = [0, 1, 2, 3, 4]
        >>> df['val'] = [float(i + 10) for i in range(5)]
        >>> df_new = df.reindex(index=[0, 3, 4, 5],
        ...                     columns=['key', 'val', 'sum'])
        >>> df
           key   val
        0    0  10.0
        1    1  11.0
        2    2  12.0
        3    3  13.0
        4    4  14.0
        >>> df_new
           key   val  sum
        0    0  10.0  NaN
        3    3  13.0  NaN
        4    4  14.0  NaN
        5   -1   NaN  NaN
        """

        if labels is None and index is None and columns is None:
            return self.copy(deep=copy)

        # pandas simply ignores the labels keyword if it is provided in
        # addition to index and columns, but it prohibits the axis arg.
        if (index is not None or columns is not None) and axis is not None:
            raise TypeError(
                "Cannot specify both 'axis' and any of 'index' or 'columns'."
            )

        axis = self._get_axis_from_axis_arg(axis)
        if axis == 0:
            if index is None:
                index = labels
        else:
            if columns is None:
                columns = labels
        df = (
            self
            if columns is None
            else self[list(set(self._column_names) & set(columns))]
        )

        return df._reindex(
            columns=columns,
            dtypes=self._dtypes,
            deep=copy,
            index=index,
            inplace=False,
        )

    def set_index(
        self,
        keys,
        drop=True,
        append=False,
        inplace=False,
        verify_integrity=False,
    ):
        """Return a new DataFrame with a new index

        Parameters
        ----------
        keys : Index, Series-convertible, label-like, or list
            Index : the new index.
            Series-convertible : values for the new index.
            Label-like : Label of column to be used as index.
            List : List of items from above.
        drop : boolean, default True
            Whether to drop corresponding column for str index argument
        append : boolean, default True
            Whether to append columns to the existing index,
            resulting in a MultiIndex.
        inplace : boolean, default False
            Modify the DataFrame in place (do not create a new object).
        verify_integrity : boolean, default False
            Check for duplicates in the new index.

        Examples
        --------
        >>> df = cudf.DataFrame({
        ...     "a": [1, 2, 3, 4, 5],
        ...     "b": ["a", "b", "c", "d","e"],
        ...     "c": [1.0, 2.0, 3.0, 4.0, 5.0]
        ... })
        >>> df
           a  b    c
        0  1  a  1.0
        1  2  b  2.0
        2  3  c  3.0
        3  4  d  4.0
        4  5  e  5.0

        Set the index to become the ‘b’ column:

        >>> df.set_index('b')
           a    c
        b
        a  1  1.0
        b  2  2.0
        c  3  3.0
        d  4  4.0
        e  5  5.0

        Create a MultiIndex using columns ‘a’ and ‘b’:

        >>> df.set_index(["a", "b"])
               c
        a b
        1 a  1.0
        2 b  2.0
        3 c  3.0
        4 d  4.0
        5 e  5.0

        Set new Index instance as index:

        >>> df.set_index(cudf.RangeIndex(10, 15))
            a  b    c
        10  1  a  1.0
        11  2  b  2.0
        12  3  c  3.0
        13  4  d  4.0
        14  5  e  5.0

        Setting `append=True` will combine current index with column `a`:

        >>> df.set_index("a", append=True)
             b    c
          a
        0 1  a  1.0
        1 2  b  2.0
        2 3  c  3.0
        3 4  d  4.0
        4 5  e  5.0

        `set_index` supports `inplace` parameter too:

        >>> df.set_index("a", inplace=True)
        >>> df
           b    c
        a
        1  a  1.0
        2  b  2.0
        3  c  3.0
        4  d  4.0
        5  e  5.0
        """

        if not isinstance(keys, list):
            keys = [keys]

        # Preliminary type check
        col_not_found = []
        columns_to_add = []
        names = []
        to_drop = []
        for col in keys:
            # Is column label
            if is_scalar(col) or isinstance(col, tuple):
                if col in self.columns:
                    columns_to_add.append(self[col])
                    names.append(col)
                    if drop:
                        to_drop.append(col)
                else:
                    col_not_found.append(col)
            else:
                # Try coerce into column
                if not is_column_like(col):
                    try:
                        col = as_column(col)
                    except TypeError:
                        msg = f"{col} cannot be converted to column-like."
                        raise TypeError(msg)
                if isinstance(col, (MultiIndex, pd.MultiIndex)):
                    col = (
                        cudf.from_pandas(col)
                        if isinstance(col, pd.MultiIndex)
                        else col
                    )
                    cols = [col._data[x] for x in col._data]
                    columns_to_add.extend(cols)
                    names.extend(col.names)
                else:
                    if isinstance(col, (pd.RangeIndex, cudf.RangeIndex)):
                        # Corner case: RangeIndex does not need to instantiate
                        columns_to_add.append(col)
                    else:
                        # For pandas obj, convert to gpu obj
                        columns_to_add.append(as_column(col))
                    if isinstance(
                        col, (cudf.Series, cudf.Index, pd.Series, pd.Index)
                    ):
                        names.append(col.name)
                    else:
                        names.append(None)

        if col_not_found:
            raise KeyError(f"None of {col_not_found} are in the columns")

        if append:
            idx_cols = [self.index._data[x] for x in self.index._data]
            if isinstance(self.index, MultiIndex):
                idx_names = self.index.names
            else:
                idx_names = [self.index.name]
            columns_to_add = idx_cols + columns_to_add
            names = idx_names + names

        if len(columns_to_add) == 0:
            raise ValueError("No valid columns to be added to index.")
        elif len(columns_to_add) == 1:
            idx = cudf.Index(columns_to_add[0], name=names[0])
        else:
            idx = MultiIndex._from_data(
                {i: col for i, col in enumerate(columns_to_add)}
            )
            idx.names = names

        if not isinstance(idx, BaseIndex):
            raise ValueError("Parameter index should be type `Index`.")

        df = self if inplace else self.copy(deep=True)

        if verify_integrity and not idx.is_unique:
            raise ValueError(f"Values in Index are not unique: {idx}")

        if to_drop:
            df.drop(columns=to_drop, inplace=True)

        df.index = idx
        return df if not inplace else None

    @docutils.doc_apply(
        doc_reset_index_template.format(
            klass="DataFrame",
            argument="",
            return_type="DataFrame or None",
            return_doc="",
            example="""
        >>> df = cudf.DataFrame([('bird', 389.0),
        ...                    ('bird', 24.0),
        ...                    ('mammal', 80.5),
        ...                    ('mammal', np.nan)],
        ...                   index=['falcon', 'parrot', 'lion', 'monkey'],
        ...                   columns=('class', 'max_speed'))
        >>> df
                 class max_speed
        falcon    bird     389.0
        parrot    bird      24.0
        lion    mammal      80.5
        monkey  mammal      <NA>
        >>> df.reset_index()
            index   class max_speed
        0  falcon    bird     389.0
        1  parrot    bird      24.0
        2    lion  mammal      80.5
        3  monkey  mammal      <NA>
        >>> df.reset_index(drop=True)
            class max_speed
        0    bird     389.0
        1    bird      24.0
        2  mammal      80.5
        3  mammal      <NA>

        You can also use ``reset_index`` with MultiIndex.

        >>> index = cudf.MultiIndex.from_tuples([('bird', 'falcon'),
        ...                                     ('bird', 'parrot'),
        ...                                     ('mammal', 'lion'),
        ...                                     ('mammal', 'monkey')],
        ...                                     names=['class', 'name'])
        >>> df = cudf.DataFrame([(389.0, 'fly'),
        ...                      ( 24.0, 'fly'),
        ...                      ( 80.5, 'run'),
        ...                      (np.nan, 'jump')],
        ...                      index=index,
        ...                      columns=('speed', 'type'))
        >>> df
                       speed  type
        class  name
        bird   falcon  389.0   fly
               parrot   24.0   fly
        mammal lion     80.5   run
               monkey   <NA>  jump
        >>> df.reset_index(level='class')
                 class  speed  type
        name
        falcon    bird  389.0   fly
        parrot    bird   24.0   fly
        lion    mammal   80.5   run
        monkey  mammal   <NA>  jump
        """,
        )
    )
    def reset_index(
        self, level=None, drop=False, inplace=False, col_level=0, col_fill=""
    ):
        return self._mimic_inplace(
            DataFrame._from_data(
                *self._reset_index(
                    level=level,
                    drop=drop,
                    col_level=col_level,
                    col_fill=col_fill,
                )
            ),
            inplace=inplace,
        )

    def take(self, indices, axis=0, keep_index=None):
        axis = self._get_axis_from_axis_arg(axis)
        if axis != 0:
            raise NotImplementedError("Only axis=0 is supported.")
        out = super().take(indices, keep_index)
        out.columns = self.columns
        return out

    @annotate("INSERT", color="green", domain="cudf_python")
    def insert(self, loc, name, value, nan_as_null=None):
        """Add a column to DataFrame at the index specified by loc.

        Parameters
        ----------
        loc : int
            location to insert by index, cannot be greater then num columns + 1
        name : number or string
            name or label of column to be inserted
        value : Series or array-like
        """
        if name in self._data:
            raise NameError(f"duplicated column name {name}")

        num_cols = len(self._data)
        if loc < 0:
            loc += num_cols + 1

        if not (0 <= loc <= num_cols):
            raise ValueError(
                f"insert location must be within range "
                f"{-(num_cols + 1) * (num_cols > 0)}, "
                f"{num_cols * (num_cols > 0)}"
            )

        if _is_scalar_or_zero_d_array(value):
            value = utils.scalar_broadcast_to(value, len(self))

        if len(self) == 0:
            if isinstance(value, (pd.Series, Series)):
                self._index = as_index(value.index)
            elif len(value) > 0:
                self._index = RangeIndex(start=0, stop=len(value))
                new_data = self._data.__class__()
                if num_cols != 0:
                    for col_name in self._data:
                        new_data[col_name] = column.column_empty_like(
                            self._data[col_name],
                            masked=True,
                            newsize=len(value),
                        )
                self._data = new_data
        elif isinstance(value, (pd.Series, Series)):
            value = Series(value, nan_as_null=nan_as_null)._align_to_index(
                self._index, how="right", sort=False
            )

        value = column.as_column(value, nan_as_null=nan_as_null)

        self._data.insert(name, value, loc=loc)

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors="raise",
    ):
        """
        Drop specified labels from rows or columns.

        Remove rows or columns by specifying label names and corresponding
        axis, or by specifying directly index or column names. When using a
        multi-index, labels on different levels can be removed by specifying
        the level.

        Parameters
        ----------
        labels : single label or list-like
            Index or column labels to drop.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Whether to drop labels from the index (0 or 'index') or
            columns (1 or 'columns').
        index : single label or list-like
            Alternative to specifying axis (``labels, axis=0``
            is equivalent to ``index=labels``).
        columns : single label or list-like
            Alternative to specifying axis (``labels, axis=1``
            is equivalent to ``columns=labels``).
        level : int or level name, optional
            For MultiIndex, level from which the labels will be removed.
        inplace : bool, default False
            If False, return a copy. Otherwise, do operation
            inplace and return None.
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and only existing labels are
            dropped.

        Returns
        -------
        DataFrame
            DataFrame without the removed index or column labels.

        Raises
        ------
        KeyError
            If any of the labels is not found in the selected axis.

        See Also
        --------
        DataFrame.loc : Label-location based indexer for selection by label.
        DataFrame.dropna : Return DataFrame with labels on given axis omitted
            where (all or any) data are missing.
        DataFrame.drop_duplicates : Return DataFrame with duplicate rows
            removed, optionally only considering certain columns.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"A": [1, 2, 3, 4],
        ...                      "B": [5, 6, 7, 8],
        ...                      "C": [10, 11, 12, 13],
        ...                      "D": [20, 30, 40, 50]})
        >>> df
           A  B   C   D
        0  1  5  10  20
        1  2  6  11  30
        2  3  7  12  40
        3  4  8  13  50

        Drop columns

        >>> df.drop(['B', 'C'], axis=1)
           A   D
        0  1  20
        1  2  30
        2  3  40
        3  4  50
        >>> df.drop(columns=['B', 'C'])
           A   D
        0  1  20
        1  2  30
        2  3  40
        3  4  50

        Drop a row by index

        >>> df.drop([0, 1])
           A  B   C   D
        2  3  7  12  40
        3  4  8  13  50

        Drop columns and/or rows of MultiIndex DataFrame

        >>> midx = cudf.MultiIndex(levels=[['lama', 'cow', 'falcon'],
        ...                              ['speed', 'weight', 'length']],
        ...                      codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2],
        ...                             [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        >>> df = cudf.DataFrame(index=midx, columns=['big', 'small'],
        ...                   data=[[45, 30], [200, 100], [1.5, 1], [30, 20],
        ...                         [250, 150], [1.5, 0.8], [320, 250],
        ...                         [1, 0.8], [0.3, 0.2]])
        >>> df
                         big  small
        lama   speed    45.0   30.0
               weight  200.0  100.0
               length    1.5    1.0
        cow    speed    30.0   20.0
               weight  250.0  150.0
               length    1.5    0.8
        falcon speed   320.0  250.0
               weight    1.0    0.8
               length    0.3    0.2
        >>> df.drop(index='cow', columns='small')
                         big
        lama   speed    45.0
               weight  200.0
               length    1.5
        falcon speed   320.0
               weight    1.0
               length    0.3
        >>> df.drop(index='length', level=1)
                         big  small
        lama   speed    45.0   30.0
               weight  200.0  100.0
        cow    speed    30.0   20.0
               weight  250.0  150.0
        falcon speed   320.0  250.0
               weight    1.0    0.8
        """

        if labels is not None:
            if index is not None or columns is not None:
                raise ValueError(
                    "Cannot specify both 'labels' and 'index'/'columns'"
                )
            target = labels
        elif index is not None:
            target = index
            axis = 0
        elif columns is not None:
            target = columns
            axis = 1
        else:
            raise ValueError(
                "Need to specify at least one of 'labels', "
                "'index' or 'columns'"
            )

        if inplace:
            out = self
        else:
            out = self.copy()

        if axis in (1, "columns"):
            target = _get_host_unique(target)

            _drop_columns(out, target, errors)
        elif axis in (0, "index"):
            dropped = _drop_rows_by_labels(out, target, level, errors)

            if columns is not None:
                columns = _get_host_unique(columns)
                _drop_columns(dropped, columns, errors)

            out._data = dropped._data
            out._index = dropped._index

        if not inplace:
            return out

    def _drop_column(self, name):
        """Drop a column by *name*"""
        if name not in self._data:
            raise KeyError(f"column '{name}' does not exist")
        del self._data[name]

    def drop_duplicates(
        self, subset=None, keep="first", inplace=False, ignore_index=False
    ):
        """
        Return DataFrame with duplicate rows removed, optionally only
        considering certain subset of columns.

        Parameters
        ----------
        subset : column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns.
        keep : {'first', 'last', False}, default 'first'
            Determines which duplicates (if any) to keep.
            - ``first`` : Drop duplicates except for the first occurrence.
            - ``last`` : Drop duplicates except for the last occurrence.
            - False : Drop all duplicates.
        inplace : bool, default False
            Whether to drop duplicates in place or to return a copy.
        ignore_index : bool, default False
            If True, the resulting axis will be labeled 0, 1, …, n - 1.

        Returns
        -------
        DataFrame or None
            DataFrame with duplicates removed or None if ``inplace=True``.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({
        ...     'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
        ...     'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
        ...     'rating': [4, 4, 3.5, 15, 5]
        ... })
        >>> df
             brand style  rating
        0  Yum Yum   cup     4.0
        1  Yum Yum   cup     4.0
        2  Indomie   cup     3.5
        3  Indomie  pack    15.0
        4  Indomie  pack     5.0

        By default, it removes duplicate rows based
        on all columns. Note that order of
        the rows being returned is not guaranteed
        to be sorted.

        >>> df.drop_duplicates()
             brand style  rating
        2  Indomie   cup     3.5
        4  Indomie  pack     5.0
        3  Indomie  pack    15.0
        0  Yum Yum   cup     4.0

        To remove duplicates on specific column(s),
        use `subset`.

        >>> df.drop_duplicates(subset=['brand'])
             brand style  rating
        2  Indomie   cup     3.5
        0  Yum Yum   cup     4.0

        To remove duplicates and keep last occurrences, use `keep`.

        >>> df.drop_duplicates(subset=['brand', 'style'], keep='last')
             brand style  rating
        2  Indomie   cup     3.5
        4  Indomie  pack     5.0
        1  Yum Yum   cup     4.0
        """  # noqa: E501
        outdf = super().drop_duplicates(
            subset=subset, keep=keep, ignore_index=ignore_index
        )

        return self._mimic_inplace(outdf, inplace=inplace)

    def pop(self, item):
        """Return a column and drop it from the DataFrame."""
        popped = self[item]
        del self[item]
        return popped

    def rename(
        self,
        mapper=None,
        index=None,
        columns=None,
        axis=0,
        copy=True,
        inplace=False,
        level=None,
        errors="ignore",
    ):
        """Alter column and index labels.

        Function / dict values must be unique (1-to-1). Labels not contained in
        a dict / Series will be left as-is. Extra labels listed don’t throw an
        error.

        ``DataFrame.rename`` supports two calling conventions:
            - ``(index=index_mapper, columns=columns_mapper, ...)``
            - ``(mapper, axis={0/'index' or 1/'column'}, ...)``

        We highly recommend using keyword arguments to clarify your intent.

        Parameters
        ----------
        mapper : dict-like or function, default None
            optional dict-like or functions transformations to apply to
            the index/column values depending on selected ``axis``.
        index : dict-like, default None
            Optional dict-like transformations to apply to the index axis'
            values. Does not support functions for axis 0 yet.
        columns : dict-like or function, default None
            optional dict-like or functions transformations to apply to
            the columns axis' values.
        axis : int, default 0
            Axis to rename with mapper.
            0 or 'index' for index
            1  or 'columns' for columns
        copy : boolean, default True
            Also copy underlying data
        inplace : boolean, default False
            Return new DataFrame.  If True, assign columns without copy
        level : int or level name, default None
            In case of a MultiIndex, only rename labels in the specified level.
        errors : {'raise', 'ignore', 'warn'}, default 'ignore'
            *Only 'ignore' supported*
            Control raising of exceptions on invalid data for provided dtype.

            -   ``raise`` : allow exceptions to be raised
            -   ``ignore`` : suppress exceptions. On error return original
                object.
            -   ``warn`` : prints last exceptions as warnings and
                return original object.

        Returns
        -------
        DataFrame

        Notes
        -----
        Difference from pandas:
            * Not supporting: level

        Rename will not overwite column names. If a list with duplicates is
        passed, column names will be postfixed with a number.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> df
           A  B
        0  1  4
        1  2  5
        2  3  6

        Rename columns using a mapping:

        >>> df.rename(columns={"A": "a", "B": "c"})
           a  c
        0  1  4
        1  2  5
        2  3  6

        Rename index using a mapping:

        >>> df.rename(index={0: 10, 1: 20, 2: 30})
            A  B
        10  1  4
        20  2  5
        30  3  6
        """
        if errors != "ignore":
            raise NotImplementedError(
                "Only errors='ignore' is currently supported"
            )

        if mapper is None and index is None and columns is None:
            return self.copy(deep=copy)

        index = mapper if index is None and axis in (0, "index") else index
        columns = (
            mapper if columns is None and axis in (1, "columns") else columns
        )

        if index:
            if (
                any(type(item) == str for item in index.values())
                and type(self.index) != cudf.StringIndex
            ):
                raise NotImplementedError(
                    "Implicit conversion of index to "
                    "mixed type is not yet supported."
                )

            if level is not None and isinstance(self.index, MultiIndex):
                out_index = self.index.copy(deep=copy)
                out_index.get_level_values(level).to_frame().replace(
                    to_replace=list(index.keys()),
                    value=list(index.values()),
                    inplace=True,
                )
                out = DataFrame(index=out_index)
            else:
                out = DataFrame(
                    index=self.index.replace(
                        to_replace=list(index.keys()),
                        value=list(index.values()),
                    )
                )
        else:
            out = DataFrame(index=self.index)

        if columns:
            out._data = self._data.rename_levels(mapper=columns, level=level)
        else:
            out._data = self._data.copy(deep=copy)

        if inplace:
            self._data = out._data
        else:
            return out.copy(deep=copy)

    def add_prefix(self, prefix):
        out = self.copy(deep=True)
        out.columns = [
            prefix + col_name for col_name in list(self._data.keys())
        ]
        return out

    def add_suffix(self, suffix):
        out = self.copy(deep=True)
        out.columns = [
            col_name + suffix for col_name in list(self._data.keys())
        ]
        return out

    def as_gpu_matrix(self, columns=None, order="F"):
        warnings.warn(
            "The as_gpu_matrix method will be removed in a future cuDF "
            "release. Consider using `to_cupy` instead.",
            FutureWarning,
        )
        if columns is None:
            columns = self._data.names

        cols = [self._data[k] for k in columns]
        ncol = len(cols)
        nrow = len(self)
        if ncol < 1:
            # This is the case for empty dataframe - construct empty cupy array
            matrix = cupy.empty(
                shape=(0, 0), dtype=cudf.dtype("float64"), order=order
            )
            return cuda.as_cuda_array(matrix)

        if any(
            (is_categorical_dtype(c) or np.issubdtype(c, cudf.dtype("object")))
            for c in cols
        ):
            raise TypeError("non-numeric data not yet supported")

        dtype = find_common_type([col.dtype for col in cols])
        for k, c in self._data.items():
            if c.has_nulls():
                raise ValueError(
                    f"column '{k}' has null values. "
                    f"hint: use .fillna() to replace null values"
                )
        cupy_dtype = dtype
        if np.issubdtype(cupy_dtype, np.datetime64):
            cupy_dtype = cudf.dtype("int64")

        if order not in ("F", "C"):
            raise ValueError(
                "order parameter should be 'C' for row major or 'F' for"
                "column major GPU matrix"
            )

        matrix = cupy.empty(shape=(nrow, ncol), dtype=cupy_dtype, order=order)
        for colidx, inpcol in enumerate(cols):
            dense = inpcol.astype(cupy_dtype)
            matrix[:, colidx] = cupy.asarray(dense)
        return cuda.as_cuda_array(matrix).view(dtype)

    def as_matrix(self, columns=None):
        warnings.warn(
            "The as_matrix method will be removed in a future cuDF "
            "release. Consider using `to_numpy` instead.",
            FutureWarning,
        )
        return self.as_gpu_matrix(columns=columns).copy_to_host()

    def label_encoding(
        self, column, prefix, cats, prefix_sep="_", dtype=None, na_sentinel=-1
    ):
        """Encode labels in a column with label encoding.

        Parameters
        ----------
        column : str
            the source column with binary encoding for the data.
        prefix : str
            the new column name prefix.
        cats : sequence of ints
            the sequence of categories as integers.
        prefix_sep : str
            the separator between the prefix and the category.
        dtype :
            the dtype for the outputs; see Series.label_encoding
        na_sentinel : number
            Value to indicate missing category.

        Returns
        -------
        A new DataFrame with a new column appended for the coded values.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a':[1, 2, 3], 'b':[10, 10, 20]})
        >>> df
           a   b
        0  1  10
        1  2  10
        2  3  20
        >>> df.label_encoding(column="b", prefix="b_col", cats=[10, 20])
           a   b  b_col_labels
        0  1  10             0
        1  2  10             0
        2  3  20             1
        """

        warnings.warn(
            "DataFrame.label_encoding is deprecated and will be removed in "
            "the future. Consider using cuML's LabelEncoder instead.",
            FutureWarning,
        )

        return self._label_encoding(
            column, prefix, cats, prefix_sep, dtype, na_sentinel
        )

    def _label_encoding(
        self, column, prefix, cats, prefix_sep="_", dtype=None, na_sentinel=-1
    ):
        # Private implementation of deprecated public label_encoding method
        newname = prefix_sep.join([prefix, "labels"])
        newcol = self[column]._label_encoding(
            cats=cats, dtype=dtype, na_sentinel=na_sentinel
        )
        outdf = self.copy()
        outdf.insert(len(outdf._data), newname, newcol)
        return outdf

    def agg(self, aggs, axis=None):
        """
        Aggregate using one or more operations over the specified axis.

        Parameters
        ----------
        aggs : Iterable (set, list, string, tuple or dict)
            Function to use for aggregating data. Accepted types are:
             * string name, e.g. ``"sum"``
             * list of functions, e.g. ``["sum", "min", "max"]``
             * dict of axis labels specified operations per column,
               e.g. ``{"a": "sum"}``

        axis : not yet supported

        Returns
        -------
        Aggregation Result : ``Series`` or ``DataFrame``
            When ``DataFrame.agg`` is called with single agg,
            ``Series`` is returned.
            When ``DataFrame.agg`` is called with several aggs,
            ``DataFrame`` is returned.

        Notes
        -----
        Difference from pandas:
          * Not supporting: ``axis``, ``*args``, ``**kwargs``

        """
        # TODO: Remove the typecasting below once issue #6846 is fixed
        # link <https://github.com/rapidsai/cudf/issues/6846>
        dtypes = [self[col].dtype for col in self._column_names]
        common_dtype = find_common_type(dtypes)
        df_normalized = self.astype(common_dtype)

        if any(is_string_dtype(dt) for dt in dtypes):
            raise NotImplementedError(
                "DataFrame.agg() is not supported for "
                "frames containing string columns"
            )

        if axis == 0 or axis is not None:
            raise NotImplementedError("axis not implemented yet")

        if isinstance(aggs, Iterable) and not isinstance(aggs, (str, dict)):
            result = DataFrame()
            # TODO : Allow simultaneous pass for multi-aggregation as
            # a future optimization
            for agg in aggs:
                result[agg] = getattr(df_normalized, agg)()
            return result.T.sort_index(axis=1, ascending=True)

        elif isinstance(aggs, str):
            if not hasattr(df_normalized, aggs):
                raise AttributeError(
                    f"{aggs} is not a valid function for "
                    f"'DataFrame' object"
                )
            result = DataFrame()
            result[aggs] = getattr(df_normalized, aggs)()
            result = result.iloc[:, 0]
            result.name = None
            return result

        elif isinstance(aggs, dict):
            cols = aggs.keys()
            if any([callable(val) for val in aggs.values()]):
                raise NotImplementedError(
                    "callable parameter is not implemented yet"
                )
            elif all([isinstance(val, str) for val in aggs.values()]):
                result = cudf.Series(index=cols)
                for key, value in aggs.items():
                    col = df_normalized[key]
                    if not hasattr(col, value):
                        raise AttributeError(
                            f"{value} is not a valid function for "
                            f"'Series' object"
                        )
                    result[key] = getattr(col, value)()
            elif all([isinstance(val, Iterable) for val in aggs.values()]):
                idxs = set()
                for val in aggs.values():
                    if isinstance(val, Iterable):
                        idxs.update(val)
                    elif isinstance(val, str):
                        idxs.add(val)
                idxs = sorted(list(idxs))
                for agg in idxs:
                    if agg is callable:
                        raise NotImplementedError(
                            "callable parameter is not implemented yet"
                        )
                result = DataFrame(index=idxs, columns=cols)
                for key in aggs.keys():
                    col = df_normalized[key]
                    col_empty = column_empty(
                        len(idxs), dtype=col.dtype, masked=True
                    )
                    ans = cudf.Series(data=col_empty, index=idxs)
                    if isinstance(aggs.get(key), Iterable):
                        # TODO : Allow simultaneous pass for multi-aggregation
                        # as a future optimization
                        for agg in aggs.get(key):
                            if not hasattr(col, agg):
                                raise AttributeError(
                                    f"{agg} is not a valid function for "
                                    f"'Series' object"
                                )
                            ans[agg] = getattr(col, agg)()
                    elif isinstance(aggs.get(key), str):
                        if not hasattr(col, aggs.get(key)):
                            raise AttributeError(
                                f"{aggs.get(key)} is not a valid function for "
                                f"'Series' object"
                            )
                        ans[aggs.get(key)] = getattr(col, agg)()
                    result[key] = ans
            else:
                raise ValueError("values of dict must be a string or list")

            return result

        elif callable(aggs):
            raise NotImplementedError(
                "callable parameter is not implemented yet"
            )

        else:
            raise ValueError("argument must be a string, list or dict")

    def nlargest(self, n, columns, keep="first"):
        """Get the rows of the DataFrame sorted by the n largest value of *columns*

        Parameters
        ----------
        n : int
            Number of rows to return.
        columns : label or list of labels
            Column label(s) to order by.
        keep : {'first', 'last'}, default 'first'
            Where there are duplicate values:

            - `first` : prioritize the first occurrence(s)
            - `last` : prioritize the last occurrence(s)

        Returns
        -------
        DataFrame
            The first `n` rows ordered by the given columns in descending
            order.

        Notes
        -----
        Difference from pandas:
            - Only a single column is supported in *columns*

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'population': [59000000, 65000000, 434000,
        ...                                   434000, 434000, 337000, 11300,
        ...                                   11300, 11300],
        ...                    'GDP': [1937894, 2583560 , 12011, 4520, 12128,
        ...                            17036, 182, 38, 311],
        ...                    'alpha-2': ["IT", "FR", "MT", "MV", "BN",
        ...                                "IS", "NR", "TV", "AI"]},
        ...                   index=["Italy", "France", "Malta",
        ...                          "Maldives", "Brunei", "Iceland",
        ...                          "Nauru", "Tuvalu", "Anguilla"])
        >>> df
                  population      GDP alpha-2
        Italy       59000000  1937894      IT
        France      65000000  2583560      FR
        Malta         434000    12011      MT
        Maldives      434000     4520      MV
        Brunei        434000    12128      BN
        Iceland       337000    17036      IS
        Nauru          11300      182      NR
        Tuvalu         11300       38      TV
        Anguilla       11300      311      AI
        >>> df.nlargest(3, 'population')
                population      GDP alpha-2
        France    65000000  2583560      FR
        Italy     59000000  1937894      IT
        Malta       434000    12011      MT
        >>> df.nlargest(3, 'population', keep='last')
                population      GDP alpha-2
        France    65000000  2583560      FR
        Italy     59000000  1937894      IT
        Brunei      434000    12128      BN
        """
        return self._n_largest_or_smallest(True, n, columns, keep)

    def nsmallest(self, n, columns, keep="first"):
        """Get the rows of the DataFrame sorted by the n smallest value of *columns*

        Parameters
        ----------
        n : int
            Number of items to retrieve.
        columns : list or str
            Column name or names to order by.
        keep : {'first', 'last'}, default 'first'
            Where there are duplicate values:

            - ``first`` : take the first occurrence.
            - ``last`` : take the last occurrence.

        Returns
        -------
        DataFrame

        Notes
        -----
        Difference from pandas:
            - Only a single column is supported in *columns*

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'population': [59000000, 65000000, 434000,
        ...                                   434000, 434000, 337000, 337000,
        ...                                   11300, 11300],
        ...                    'GDP': [1937894, 2583560 , 12011, 4520, 12128,
        ...                            17036, 182, 38, 311],
        ...                    'alpha-2': ["IT", "FR", "MT", "MV", "BN",
        ...                                "IS", "NR", "TV", "AI"]},
        ...                   index=["Italy", "France", "Malta",
        ...                          "Maldives", "Brunei", "Iceland",
        ...                          "Nauru", "Tuvalu", "Anguilla"])
        >>> df
                  population      GDP alpha-2
        Italy       59000000  1937894      IT
        France      65000000  2583560      FR
        Malta         434000    12011      MT
        Maldives      434000     4520      MV
        Brunei        434000    12128      BN
        Iceland       337000    17036      IS
        Nauru         337000      182      NR
        Tuvalu         11300       38      TV
        Anguilla       11300      311      AI

        In the following example, we will use ``nsmallest`` to select the
        three rows having the smallest values in column "population".

        >>> df.nsmallest(3, 'population')
                  population    GDP alpha-2
        Tuvalu         11300     38      TV
        Anguilla       11300    311      AI
        Iceland       337000  17036      IS

        When using ``keep='last'``, ties are resolved in reverse order:

        >>> df.nsmallest(3, 'population', keep='last')
                  population  GDP alpha-2
        Anguilla       11300  311      AI
        Tuvalu         11300   38      TV
        Nauru         337000  182      NR
        """
        return self._n_largest_or_smallest(False, n, columns, keep)

    def transpose(self):
        """Transpose index and columns.

        Returns
        -------
        a new (ncol x nrow) dataframe. self is (nrow x ncol)

        Notes
        -----
        Difference from pandas:
        Not supporting *copy* because default and only behavior is copy=True
        """
        # Never transpose a MultiIndex - remove the existing columns and
        # replace with a RangeIndex. Afterward, reassign.
        columns = self.index.copy(deep=False)
        index = self.columns.copy(deep=False)
        if self._num_columns == 0 or self._num_rows == 0:
            return DataFrame(index=index, columns=columns)
        # Set the old column names as the new index
        result = self.__class__._from_data(
            # Cython renames the columns to the range [0...ncols]
            libcudf.transpose.transpose(self),
            as_index(index),
        )
        # Set the old index as the new column names
        result.columns = columns
        return result

    T = property(transpose, doc=transpose.__doc__)

    def melt(self, **kwargs):
        """Unpivots a DataFrame from wide format to long format,
        optionally leaving identifier variables set.

        Parameters
        ----------
        frame : DataFrame
        id_vars : tuple, list, or ndarray, optional
            Column(s) to use as identifier variables.
            default: None
        value_vars : tuple, list, or ndarray, optional
            Column(s) to unpivot.
            default: all columns that are not set as `id_vars`.
        var_name : scalar
            Name to use for the `variable` column.
            default: frame.columns.name or 'variable'
        value_name : str
            Name to use for the `value` column.
            default: 'value'

        Returns
        -------
        out : DataFrame
            Melted result
        """
        from cudf.core.reshape import melt

        return melt(self, **kwargs)

    @annotate("JOIN", color="blue", domain="cudf_python")
    def merge(
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
        method=None,
        indicator=False,
        suffixes=("_x", "_y"),
    ):
        """Merge GPU DataFrame objects by performing a database-style join
        operation by columns or indexes.

        Parameters
        ----------
        right : DataFrame
        on : label or list; defaults to None
            Column or index level names to join on. These must be found in
            both DataFrames.

            If on is None and not merging on indexes then
            this defaults to the intersection of the columns
            in both DataFrames.
        how : {‘left’, ‘outer’, ‘inner’, 'leftsemi', 'leftanti'}, \
            default ‘inner’
            Type of merge to be performed.

            - left : use only keys from left frame, similar to a SQL left
              outer join.
            - right : not supported.
            - outer : use union of keys from both frames, similar to a SQL
              full outer join.
            - inner : use intersection of keys from both frames, similar to
              a SQL inner join.
            - leftsemi : similar to ``inner`` join, but only returns columns
               from the left dataframe and ignores all columns from the
               right dataframe.
            - leftanti : returns only rows columns from the left dataframe
              for non-matched records. This is exact opposite to ``leftsemi``
              join.
        left_on : label or list, or array-like
            Column or index level names to join on in the left DataFrame.
            Can also be an array or list of arrays of the length of the
            left DataFrame. These arrays are treated as if they are columns.
        right_on : label or list, or array-like
            Column or index level names to join on in the right DataFrame.
            Can also be an array or list of arrays of the length of the
            right DataFrame. These arrays are treated as if they are columns.
        left_index : bool, default False
            Use the index from the left DataFrame as the join key(s).
        right_index : bool, default False
            Use the index from the right DataFrame as the join key.
        sort : bool, default False
            Sort the resulting dataframe by the columns that were merged on,
            starting from the left.
        suffixes: Tuple[str, str], defaults to ('_x', '_y')
            Suffixes applied to overlapping column names on the left and right
            sides
        method :
            This parameter is unused. It is deprecated and will be removed in a
            future version.

        Returns
        -------
            merged : DataFrame

        Notes
        -----
        **DataFrames merges in cuDF result in non-deterministic row ordering.**

        Examples
        --------
        >>> import cudf
        >>> df_a = cudf.DataFrame()
        >>> df_a['key'] = [0, 1, 2, 3, 4]
        >>> df_a['vals_a'] = [float(i + 10) for i in range(5)]
        >>> df_b = cudf.DataFrame()
        >>> df_b['key'] = [1, 2, 4]
        >>> df_b['vals_b'] = [float(i+10) for i in range(3)]
        >>> df_merged = df_a.merge(df_b, on=['key'], how='left')
        >>> df_merged.sort_values('key')  # doctest: +SKIP
           key  vals_a  vals_b
        3    0    10.0
        0    1    11.0    10.0
        1    2    12.0    11.0
        4    3    13.0
        2    4    14.0    12.0

        **Merging on categorical variables is only allowed in certain cases**

        Categorical variable typecasting logic depends on both `how`
        and the specifics of the categorical variables to be merged.
        Merging categorical variables when only one side is ordered
        is ambiguous and not allowed. Merging when both categoricals
        are ordered is allowed, but only when the categories are
        exactly equal and have equal ordering, and will result in the
        common dtype.
        When both sides are unordered, the result categorical depends
        on the kind of join:
        - For inner joins, the result will be the intersection of the
        categories
        - For left or right joins, the result will be the the left or
        right dtype respectively. This extends to semi and anti joins.
        - For outer joins, the result will be the union of categories
        from both sides.
        """
        if indicator:
            raise NotImplementedError(
                "Only indicator=False is currently supported"
            )

        if lsuffix or rsuffix:
            raise ValueError(
                "The lsuffix and rsuffix keywords have been replaced with the "
                "``suffixes=`` keyword.  "
                "Please provide the following instead: \n\n"
                "    suffixes=('%s', '%s')"
                % (lsuffix or "_x", rsuffix or "_y")
            )
        else:
            lsuffix, rsuffix = suffixes

        if method is not None:
            warnings.warn(
                "The 'method' argument is deprecated and will be removed "
                "in a future version of cudf.",
                FutureWarning,
            )

        # Compute merge
        gdf_result = super()._merge(
            right,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            sort=sort,
            indicator=indicator,
            suffixes=suffixes,
        )
        return gdf_result

    @annotate("JOIN", color="blue", domain="cudf_python")
    def join(
        self,
        other,
        on=None,
        how="left",
        lsuffix="",
        rsuffix="",
        sort=False,
        method=None,
    ):
        """Join columns with other DataFrame on index or on a key column.

        Parameters
        ----------
        other : DataFrame
        how : str
            Only accepts "left", "right", "inner", "outer"
        lsuffix, rsuffix : str
            The suffices to add to the left (*lsuffix*) and right (*rsuffix*)
            column names when avoiding conflicts.
        sort : bool
            Set to True to ensure sorted ordering.
        method :
            This parameter is unused. It is deprecated and will be removed in a
            future version.

        Returns
        -------
        joined : DataFrame

        Notes
        -----
        Difference from pandas:

        - *other* must be a single DataFrame for now.
        - *on* is not supported yet due to lack of multi-index support.
        """

        if method is not None:
            warnings.warn(
                "The 'method' argument is deprecated and will be removed "
                "in a future version of cudf.",
                FutureWarning,
            )

        df = self.merge(
            other,
            left_index=True,
            right_index=True,
            how=how,
            suffixes=(lsuffix, rsuffix),
            sort=sort,
        )
        df.index.name = (
            None if self.index.name != other.index.name else self.index.name
        )
        return df

    @copy_docstring(DataFrameGroupBy)
    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=False,
        group_keys=True,
        squeeze=False,
        observed=False,
        dropna=True,
    ):
        if axis not in (0, "index"):
            raise NotImplementedError("axis parameter is not yet implemented")

        if group_keys is not True:
            raise NotImplementedError(
                "The group_keys keyword is not yet implemented"
            )

        if squeeze is not False:
            raise NotImplementedError(
                "squeeze parameter is not yet implemented"
            )

        if observed is not False:
            raise NotImplementedError(
                "observed parameter is not yet implemented"
            )

        if by is None and level is None:
            raise TypeError(
                "groupby() requires either by or level to be specified."
            )

        return (
            DataFrameResampler(self, by=by)
            if isinstance(by, cudf.Grouper) and by.freq
            else DataFrameGroupBy(
                self,
                by=by,
                level=level,
                as_index=as_index,
                dropna=dropna,
                sort=sort,
            )
        )

    def query(self, expr, local_dict=None):
        """
        Query with a boolean expression using Numba to compile a GPU kernel.

        See pandas.DataFrame.query.

        Parameters
        ----------

        expr : str
            A boolean expression. Names in expression refer to columns.
            `index` can be used instead of index name, but this is not
            supported for MultiIndex.

            Names starting with `@` refer to Python variables.

            An output value will be `null` if any of the input values are
            `null` regardless of expression.

        local_dict : dict
            Containing the local variable to be used in query.

        Returns
        -------

        filtered :  DataFrame

        Examples
        --------
        >>> import cudf
        >>> a = ('a', [1, 2, 2])
        >>> b = ('b', [3, 4, 5])
        >>> df = cudf.DataFrame([a, b])
        >>> expr = "(a == 2 and b == 4) or (b == 3)"
        >>> df.query(expr)
           a  b
        0  1  3
        1  2  4

        DateTime conditionals:

        >>> import numpy as np
        >>> import datetime
        >>> df = cudf.DataFrame()
        >>> data = np.array(['2018-10-07', '2018-10-08'], dtype='datetime64')
        >>> df['datetimes'] = data
        >>> search_date = datetime.datetime.strptime('2018-10-08', '%Y-%m-%d')
        >>> df.query('datetimes==@search_date')
                        datetimes
        1 2018-10-08T00:00:00.000

        Using local_dict:

        >>> import numpy as np
        >>> import datetime
        >>> df = cudf.DataFrame()
        >>> data = np.array(['2018-10-07', '2018-10-08'], dtype='datetime64')
        >>> df['datetimes'] = data
        >>> search_date2 = datetime.datetime.strptime('2018-10-08', '%Y-%m-%d')
        >>> df.query('datetimes==@search_date',
        ...         local_dict={'search_date':search_date2})
                        datetimes
        1 2018-10-08T00:00:00.000
        """
        # can't use `annotate` decorator here as we inspect the calling
        # environment.
        with annotate("QUERY", color="purple", domain="cudf_python"):
            if local_dict is None:
                local_dict = {}

            if self.empty:
                return self.copy()

            if not isinstance(local_dict, dict):
                raise TypeError(
                    f"local_dict type: expected dict but found "
                    f"{type(local_dict)}"
                )

            # Get calling environment
            callframe = inspect.currentframe().f_back
            callenv = {
                "locals": callframe.f_locals,
                "globals": callframe.f_globals,
                "local_dict": local_dict,
            }
            # Run query
            boolmask = queryutils.query_execute(self, expr, callenv)
            return self._apply_boolean_mask(boolmask)

    def apply(
        self, func, axis=1, raw=False, result_type=None, args=(), **kwargs
    ):
        """
        Apply a function along an axis of the DataFrame.

        Designed to mimic `pandas.DataFrame.apply`. Applies a user
        defined function row wise over a dataframe, with true null
        handling. Works with UDFs using `core.udf.pipeline.nulludf`
        and returns a single series. Uses numba to jit compile the
        function to PTX via LLVM.

        Parameters
        ----------
        func : function
            Function to apply to each row.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Axis along which the function is applied:
            * 0 or 'index': apply function to each column.
              Note: axis=0 is not yet supported.
            * 1 or 'columns': apply function to each row.
        raw: bool, default False
            Not yet supported
        result_type: {'expand', 'reduce', 'broadcast', None}, default None
            Not yet supported
        args: tuple
            Positional arguments to pass to func in addition to the dataframe.

        Examples
        --------

        Simple function of a single variable which could be NA

        >>> def f(row):
        ...     if row['a'] is cudf.NA:
        ...             return 0
        ...     else:
        ...             return row['a'] + 1
        ...
        >>> df = cudf.DataFrame({'a': [1, cudf.NA, 3]})
        >>> df.apply(f, axis=1)
        0    2
        1    0
        2    4
        dtype: int64

        Function of multiple variables will operate in
        a null aware manner

        >>> def f(row):
        ...     return row['a'] - row['b']
        ...
        >>> df = cudf.DataFrame({
        ...     'a': [1, cudf.NA, 3, cudf.NA],
        ...     'b': [5, 6, cudf.NA, cudf.NA]
        ... })
        >>> df.apply(f)
        0      -4
        1    <NA>
        2    <NA>
        3    <NA>
        dtype: int64

        Functions may conditionally return NA as in pandas

        >>> def f(row):
        ...     if row['a'] + row['b'] > 3:
        ...             return cudf.NA
        ...     else:
        ...             return row['a'] + row['b']
        ...
        >>> df = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [2, 1, 1]
        ... })
        >>> df.apply(f, axis=1)
        0       3
        1       3
        2    <NA>
        dtype: int64

        Mixed types are allowed, but will return the common
        type, rather than object as in pandas

        >>> def f(row):
        ...     return row['a'] + row['b']
        ...
        >>> df = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [0.5, cudf.NA, 3.14]
        ... })
        >>> df.apply(f, axis=1)
        0     1.5
        1    <NA>
        2    6.14
        dtype: float64

        Functions may also return scalar values, however the
        result will be promoted to a safe type regardless of
        the data

        >>> def f(row):
        ...     if row['a'] > 3:
        ...             return row['a']
        ...     else:
        ...             return 1.5
        ...
        >>> df = cudf.DataFrame({
        ...     'a': [1, 3, 5]
        ... })
        >>> df.apply(f, axis=1)
        0    1.5
        1    1.5
        2    5.0
        dtype: float64

        Ops against N columns are supported generally

        >>> def f(row):
        ...     v, w, x, y, z = (
        ...         row['a'], row['b'], row['c'], row['d'], row['e']
        ...     )
        ...     return x + (y - (z / w)) % v
        ...
        >>> df = cudf.DataFrame({
        ...     'a': [1, 2, 3],
        ...     'b': [4, 5, 6],
        ...     'c': [cudf.NA, 4, 4],
        ...     'd': [8, 7, 8],
        ...     'e': [7, 1, 6]
        ... })
        >>> df.apply(f, axis=1)
        0    <NA>
        1     4.8
        2     5.0
        dtype: float64
        """
        if axis != 1:
            raise ValueError(
                "DataFrame.apply currently only supports row wise ops"
            )
        if raw:
            raise ValueError("The `raw` kwarg is not yet supported.")
        if result_type is not None:
            raise ValueError("The `result_type` kwarg is not yet supported.")
        if kwargs:
            raise ValueError("UDFs using **kwargs are not yet supported.")

        return self._apply(func, *args)

    @applyutils.doc_apply()
    def apply_rows(
        self,
        func,
        incols,
        outcols,
        kwargs,
        pessimistic_nulls=True,
        cache_key=None,
    ):
        """
        Apply a row-wise user defined function.

        Parameters
        ----------
        {params}

        Examples
        --------
        The user function should loop over the columns and set the output for
        each row. Loop execution order is arbitrary, so each iteration of
        the loop **MUST** be independent of each other.

        When ``func`` is invoked, the array args corresponding to the
        input/output are strided so as to improve GPU parallelism.
        The loop in the function resembles serial code, but executes
        concurrently in multiple threads.

        >>> import cudf
        >>> import numpy as np
        >>> df = cudf.DataFrame()
        >>> nelem = 3
        >>> df['in1'] = np.arange(nelem)
        >>> df['in2'] = np.arange(nelem)
        >>> df['in3'] = np.arange(nelem)

        Define input columns for the kernel

        >>> in1 = df['in1']
        >>> in2 = df['in2']
        >>> in3 = df['in3']
        >>> def kernel(in1, in2, in3, out1, out2, kwarg1, kwarg2):
        ...     for i, (x, y, z) in enumerate(zip(in1, in2, in3)):
        ...         out1[i] = kwarg2 * x - kwarg1 * y
        ...         out2[i] = y - kwarg1 * z

        Call ``.apply_rows`` with the name of the input columns, the name and
        dtype of the output columns, and, optionally, a dict of extra
        arguments.

        >>> df.apply_rows(kernel,
        ...               incols=['in1', 'in2', 'in3'],
        ...               outcols=dict(out1=np.float64, out2=np.float64),
        ...               kwargs=dict(kwarg1=3, kwarg2=4))
           in1  in2  in3 out1 out2
        0    0    0    0  0.0  0.0
        1    1    1    1  1.0 -2.0
        2    2    2    2  2.0 -4.0
        """
        for col in incols:
            current_col_dtype = self._data[col].dtype
            if is_string_dtype(current_col_dtype) or is_categorical_dtype(
                current_col_dtype
            ):
                raise TypeError(
                    "User defined functions are currently not "
                    "supported on Series with dtypes `str` and `category`."
                )
        return applyutils.apply_rows(
            self,
            func,
            incols,
            outcols,
            kwargs,
            pessimistic_nulls,
            cache_key=cache_key,
        )

    @applyutils.doc_applychunks()
    def apply_chunks(
        self,
        func,
        incols,
        outcols,
        kwargs=None,
        pessimistic_nulls=True,
        chunks=None,
        blkct=None,
        tpb=None,
    ):
        """
        Transform user-specified chunks using the user-provided function.

        Parameters
        ----------
        {params}
        {params_chunks}

        Examples
        --------

        For ``tpb > 1``, ``func`` is executed by ``tpb`` number of threads
        concurrently.  To access the thread id and count,
        use ``numba.cuda.threadIdx.x`` and ``numba.cuda.blockDim.x``,
        respectively (See `numba CUDA kernel documentation`_).

        .. _numba CUDA kernel documentation:\
        http://numba.pydata.org/numba-doc/latest/cuda/kernels.html

        In the example below, the *kernel* is invoked concurrently on each
        specified chunk. The *kernel* computes the corresponding output
        for the chunk.

        By looping over the range
        ``range(cuda.threadIdx.x, in1.size, cuda.blockDim.x)``, the *kernel*
        function can be used with any *tpb* in an efficient manner.

        >>> from numba import cuda
        >>> @cuda.jit
        ... def kernel(in1, in2, in3, out1):
        ...      for i in range(cuda.threadIdx.x, in1.size, cuda.blockDim.x):
        ...          x = in1[i]
        ...          y = in2[i]
        ...          z = in3[i]
        ...          out1[i] = x * y + z

        See also
        --------
        DataFrame.apply_rows
        """
        if kwargs is None:
            kwargs = {}
        if chunks is None:
            raise ValueError("*chunks* must be defined")
        return applyutils.apply_chunks(
            self,
            func,
            incols,
            outcols,
            kwargs,
            pessimistic_nulls,
            chunks,
            tpb=tpb,
        )

    def partition_by_hash(self, columns, nparts, keep_index=True):
        """Partition the dataframe by the hashed value of data in *columns*.

        Parameters
        ----------
        columns : sequence of str
            The names of the columns to be hashed.
            Must have at least one name.
        nparts : int
            Number of output partitions
        keep_index : boolean
            Whether to keep the index or drop it

        Returns
        -------
        partitioned: list of DataFrame
        """
        idx = (
            0
            if (self._index is None or keep_index is False)
            else self._index._num_columns
        )
        key_indices = [self._data.names.index(k) + idx for k in columns]

        output_data, output_index, offsets = libcudf.hash.hash_partition(
            self, key_indices, nparts, keep_index
        )
        outdf = self.__class__._from_data(output_data, output_index)
        outdf._copy_type_metadata(self, include_index=keep_index)

        # Slice into partition
        return [outdf[s:e] for s, e in zip(offsets, offsets[1:] + [None])]

    def info(
        self,
        verbose=None,
        buf=None,
        max_cols=None,
        memory_usage=None,
        null_counts=None,
    ):
        """
        Print a concise summary of a DataFrame.

        This method prints information about a DataFrame including
        the index dtype and column dtypes, non-null values and memory usage.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print the full summary. By default, the setting in
            ``pandas.options.display.max_info_columns`` is followed.
        buf : writable buffer, defaults to sys.stdout
            Where to send the output. By default, the output is printed to
            sys.stdout. Pass a writable buffer if you need to further process
            the output.
        max_cols : int, optional
            When to switch from the verbose to the truncated output. If the
            DataFrame has more than `max_cols` columns, the truncated output
            is used. By default, the setting in
            ``pandas.options.display.max_info_columns`` is used.
        memory_usage : bool, str, optional
            Specifies whether total memory usage of the DataFrame
            elements (including the index) should be displayed. By default,
            this follows the ``pandas.options.display.memory_usage`` setting.
            True always show memory usage. False never shows memory usage.
            A value of 'deep' is equivalent to "True with deep introspection".
            Memory usage is shown in human-readable units (base-2
            representation). Without deep introspection a memory estimation is
            made based in column dtype and number of rows assuming values
            consume the same memory amount for corresponding dtypes. With deep
            memory introspection, a real memory usage calculation is performed
            at the cost of computational resources.
        null_counts : bool, optional
            Whether to show the non-null counts. By default, this is shown
            only if the frame is smaller than
            ``pandas.options.display.max_info_rows`` and
            ``pandas.options.display.max_info_columns``. A value of True always
            shows the counts, and False never shows the counts.

        Returns
        -------
        None
            This method prints a summary of a DataFrame and returns None.

        See Also
        --------
        DataFrame.describe: Generate descriptive statistics of DataFrame
            columns.
        DataFrame.memory_usage: Memory usage of DataFrame columns.

        Examples
        --------
        >>> import cudf
        >>> int_values = [1, 2, 3, 4, 5]
        >>> text_values = ['alpha', 'beta', 'gamma', 'delta', 'epsilon']
        >>> float_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        >>> df = cudf.DataFrame({"int_col": int_values,
        ...                     "text_col": text_values,
        ...                     "float_col": float_values})
        >>> df
           int_col text_col  float_col
        0        1    alpha       0.00
        1        2     beta       0.25
        2        3    gamma       0.50
        3        4    delta       0.75
        4        5  epsilon       1.00

        Prints information of all columns:

        >>> df.info(verbose=True)
        <class 'cudf.core.dataframe.DataFrame'>
        RangeIndex: 5 entries, 0 to 4
        Data columns (total 3 columns):
         #   Column     Non-Null Count  Dtype
        ---  ------     --------------  -----
         0   int_col    5 non-null      int64
         1   text_col   5 non-null      object
         2   float_col  5 non-null      float64
        dtypes: float64(1), int64(1), object(1)
        memory usage: 130.0+ bytes

        Prints a summary of columns count and its dtypes but not per column
        information:

        >>> df.info(verbose=False)
        <class 'cudf.core.dataframe.DataFrame'>
        RangeIndex: 5 entries, 0 to 4
        Columns: 3 entries, int_col to float_col
        dtypes: float64(1), int64(1), object(1)
        memory usage: 130.0+ bytes

        Pipe output of DataFrame.info to buffer instead of sys.stdout,
        get buffer content and writes to a text file:

        >>> import io
        >>> buffer = io.StringIO()
        >>> df.info(buf=buffer)
        >>> s = buffer.getvalue()
        >>> with open("df_info.txt", "w",
        ...           encoding="utf-8") as f:
        ...     f.write(s)
        ...
        369

        The `memory_usage` parameter allows deep introspection mode, specially
        useful for big DataFrames and fine-tune memory optimization:

        >>> import numpy as np
        >>> random_strings_array = np.random.choice(['a', 'b', 'c'], 10 ** 6)
        >>> df = cudf.DataFrame({
        ...     'column_1': np.random.choice(['a', 'b', 'c'], 10 ** 6),
        ...     'column_2': np.random.choice(['a', 'b', 'c'], 10 ** 6),
        ...     'column_3': np.random.choice(['a', 'b', 'c'], 10 ** 6)
        ... })
        >>> df.info(memory_usage='deep')
        <class 'cudf.core.dataframe.DataFrame'>
        RangeIndex: 1000000 entries, 0 to 999999
        Data columns (total 3 columns):
         #   Column    Non-Null Count    Dtype
        ---  ------    --------------    -----
         0   column_1  1000000 non-null  object
         1   column_2  1000000 non-null  object
         2   column_3  1000000 non-null  object
        dtypes: object(3)
        memory usage: 14.3 MB
        """
        if buf is None:
            buf = sys.stdout

        lines = [str(type(self))]

        index_name = type(self._index).__name__
        if len(self._index) > 0:
            entries_summary = f", {self._index[0]} to {self._index[-1]}"
        else:
            entries_summary = ""
        index_summary = (
            f"{index_name}: {len(self._index)} entries{entries_summary}"
        )
        lines.append(index_summary)

        if len(self.columns) == 0:
            lines.append(f"Empty {type(self).__name__}")
            cudf.utils.ioutils.buffer_write_lines(buf, lines)
            return

        cols = self.columns
        col_count = len(self.columns)

        if max_cols is None:
            max_cols = pd.options.display.max_info_columns

        max_rows = pd.options.display.max_info_rows

        if null_counts is None:
            show_counts = (col_count <= max_cols) and (len(self) < max_rows)
        else:
            show_counts = null_counts

        exceeds_info_cols = col_count > max_cols

        def _put_str(s, space):
            return str(s)[:space].ljust(space)

        def _verbose_repr():
            lines.append(f"Data columns (total {len(self.columns)} columns):")

            id_head = " # "
            column_head = "Column"
            col_space = 2

            max_col = max(len(pprint_thing(k)) for k in cols)
            len_column = len(pprint_thing(column_head))
            space = max(max_col, len_column) + col_space

            max_id = len(pprint_thing(col_count))
            len_id = len(pprint_thing(id_head))
            space_num = max(max_id, len_id) + col_space
            counts = None

            header = _put_str(id_head, space_num) + _put_str(
                column_head, space
            )
            if show_counts:
                counts = self.count().to_pandas().tolist()
                if len(cols) != len(counts):
                    raise AssertionError(
                        f"Columns must equal "
                        f"counts ({len(cols)} != {len(counts)})"
                    )
                count_header = "Non-Null Count"
                len_count = len(count_header)
                non_null = " non-null"
                max_count = max(len(pprint_thing(k)) for k in counts) + len(
                    non_null
                )
                space_count = max(len_count, max_count) + col_space
                count_temp = "{count}" + non_null
            else:
                count_header = ""
                space_count = len(count_header)
                len_count = space_count
                count_temp = "{count}"

            dtype_header = "Dtype"
            len_dtype = len(dtype_header)
            max_dtypes = max(len(pprint_thing(k)) for k in self.dtypes)
            space_dtype = max(len_dtype, max_dtypes)
            header += (
                _put_str(count_header, space_count)
                + _put_str(dtype_header, space_dtype).rstrip()
            )

            lines.append(header)
            lines.append(
                _put_str("-" * len_id, space_num)
                + _put_str("-" * len_column, space)
                + _put_str("-" * len_count, space_count)
                + _put_str("-" * len_dtype, space_dtype).rstrip()
            )

            for i, col in enumerate(self.columns):
                dtype = self.dtypes.iloc[i]
                col = pprint_thing(col)

                line_no = _put_str(" {num}".format(num=i), space_num)
                count = ""
                if show_counts:
                    count = counts[i]

                lines.append(
                    line_no
                    + _put_str(col, space)
                    + _put_str(count_temp.format(count=count), space_count)
                    + _put_str(dtype, space_dtype).rstrip()
                )

        def _non_verbose_repr():
            if len(self.columns) > 0:
                entries_summary = f", {self.columns[0]} to {self.columns[-1]}"
            else:
                entries_summary = ""
            columns_summary = (
                f"Columns: {len(self.columns)} entries{entries_summary}"
            )
            lines.append(columns_summary)

        def _sizeof_fmt(num, size_qualifier):
            # returns size in human readable format
            for x in ["bytes", "KB", "MB", "GB", "TB"]:
                if num < 1024.0:
                    return f"{num:3.1f}{size_qualifier} {x}"
                num /= 1024.0
            return f"{num:3.1f}{size_qualifier} PB"

        if verbose:
            _verbose_repr()
        elif verbose is False:  # specifically set to False, not nesc None
            _non_verbose_repr()
        else:
            if exceeds_info_cols:
                _non_verbose_repr()
            else:
                _verbose_repr()

        dtype_counts = defaultdict(int)
        for col in self._data:
            dtype_counts[self._data[col].dtype.name] += 1

        dtypes = [f"{k[0]}({k[1]:d})" for k in sorted(dtype_counts.items())]
        lines.append(f"dtypes: {', '.join(dtypes)}")

        if memory_usage is None:
            memory_usage = pd.options.display.memory_usage

        if memory_usage:
            # append memory usage of df to display
            size_qualifier = ""
            if memory_usage == "deep":
                deep = True
            else:
                deep = False
                if "object" in dtype_counts or self.index.dtype == "object":
                    size_qualifier = "+"
            mem_usage = self.memory_usage(index=True, deep=deep).sum()
            lines.append(
                f"memory usage: {_sizeof_fmt(mem_usage, size_qualifier)}\n"
            )

        cudf.utils.ioutils.buffer_write_lines(buf, lines)

    @docutils.doc_describe()
    def describe(
        self,
        percentiles=None,
        include=None,
        exclude=None,
        datetime_is_numeric=False,
    ):
        """{docstring}"""

        if not include and not exclude:
            default_include = [np.number]
            if datetime_is_numeric:
                default_include.append("datetime")
            data_to_describe = self.select_dtypes(include=default_include)
            if len(data_to_describe.columns) == 0:
                data_to_describe = self

        elif include == "all":
            if exclude is not None:
                raise ValueError("exclude must be None when include is 'all'")

            data_to_describe = self
        else:
            data_to_describe = self.select_dtypes(
                include=include, exclude=exclude
            )

            if data_to_describe.empty:
                raise ValueError("No data of included types.")

        describe_series_list = [
            data_to_describe[col].describe(percentiles=percentiles)
            for col in data_to_describe.columns
        ]
        if len(describe_series_list) == 1:
            return describe_series_list[0].to_frame()
        else:
            ldesc_indexes = sorted(
                (x.index for x in describe_series_list), key=len
            )
            names = dict.fromkeys(
                [
                    name
                    for idxnames in ldesc_indexes
                    for name in idxnames.to_pandas()
                ],
                None,
            )

            return cudf.concat(
                [
                    series.reindex(names, copy=False)
                    for series in describe_series_list
                ],
                axis=1,
                sort=False,
            )

    def to_pandas(self, nullable=False, **kwargs):
        """
        Convert to a Pandas DataFrame.

        Parameters
        ----------
        nullable : Boolean, Default False
            If ``nullable`` is ``True``, the resulting columns
            in the dataframe will be having a corresponding
            nullable Pandas dtype. If ``nullable`` is ``False``,
            the resulting columns will either convert null
            values to ``np.nan`` or ``None`` depending on the dtype.

        Returns
        -------
        out : Pandas DataFrame

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [0, 1, 2], 'b': [-3, 2, 0]})
        >>> pdf = df.to_pandas()
        >>> pdf
           a  b
        0  0 -3
        1  1  2
        2  2  0
        >>> type(pdf)
        <class 'pandas.core.frame.DataFrame'>

        ``nullable`` parameter can be used to control
        whether dtype can be Pandas Nullable or not:

        >>> df = cudf.DataFrame({'a': [0, None, 2], 'b': [True, False, None]})
        >>> df
              a      b
        0     0   True
        1  <NA>  False
        2     2   <NA>
        >>> pdf = df.to_pandas(nullable=True)
        >>> pdf
              a      b
        0     0   True
        1  <NA>  False
        2     2   <NA>
        >>> pdf.dtypes
        a      Int64
        b    boolean
        dtype: object
        >>> pdf = df.to_pandas(nullable=False)
        >>> pdf
             a      b
        0  0.0   True
        1  NaN  False
        2  2.0   None
        >>> pdf.dtypes
        a    float64
        b     object
        dtype: object
        """
        out_data = {}
        out_index = self.index.to_pandas()

        if not isinstance(self.columns, pd.Index):
            out_columns = self.columns.to_pandas()
        else:
            out_columns = self.columns

        for i, col_key in enumerate(self._data):
            out_data[i] = self._data[col_key].to_pandas(
                index=out_index, nullable=nullable
            )

        if isinstance(self.columns, BaseIndex):
            out_columns = self.columns.to_pandas()
            if isinstance(self.columns, MultiIndex):
                if self.columns.names is not None:
                    out_columns.names = self.columns.names
            else:
                out_columns.name = self.columns.name

        out_df = pd.DataFrame(out_data, index=out_index)
        out_df.columns = out_columns
        return out_df

    @classmethod
    def from_pandas(cls, dataframe, nan_as_null=None):
        """
        Convert from a Pandas DataFrame.

        Parameters
        ----------
        dataframe : Pandas DataFrame object
            A Pandads DataFrame object which has to be converted
            to cuDF DataFrame.
        nan_as_null : bool, Default True
            If ``True``, converts ``np.nan`` values to ``null`` values.
            If ``False``, leaves ``np.nan`` values as is.

        Raises
        ------
        TypeError for invalid input type.

        Examples
        --------
        >>> import cudf
        >>> import pandas as pd
        >>> data = [[0,1], [1,2], [3,4]]
        >>> pdf = pd.DataFrame(data, columns=['a', 'b'], dtype=int)
        >>> cudf.from_pandas(pdf)
           a  b
        0  0  1
        1  1  2
        2  3  4
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("not a pandas.DataFrame")

        if not dataframe.columns.is_unique:
            raise ValueError("Duplicate column names are not allowed")

        df = cls()
        # Set columns
        for col_name, col_value in dataframe.iteritems():
            # necessary because multi-index can return multiple
            # columns for a single key
            if len(col_value.shape) == 1:
                df[col_name] = column.as_column(
                    col_value.array, nan_as_null=nan_as_null
                )
            else:
                vals = col_value.values.T
                if vals.shape[0] == 1:
                    df[col_name] = column.as_column(
                        vals.flatten(), nan_as_null=nan_as_null
                    )
                else:
                    if isinstance(col_name, tuple):
                        col_name = str(col_name)
                    for idx in range(len(vals.shape)):
                        df[col_name] = column.as_column(
                            vals[idx], nan_as_null=nan_as_null
                        )

        # Set columns only if it is a MultiIndex
        if isinstance(dataframe.columns, pd.MultiIndex):
            df.columns = dataframe.columns

        # Set index
        index = cudf.from_pandas(dataframe.index, nan_as_null=nan_as_null)
        result = df.set_index(index)

        return result

    @classmethod
    def from_arrow(cls, table):
        """
        Convert from PyArrow Table to DataFrame.

        Parameters
        ----------
        table : PyArrow Table Object
            PyArrow Table Object which has to be converted to cudf DataFrame.

        Raises
        ------
        TypeError for invalid input type.

        Returns
        -------
        cudf DataFrame

        Notes
        -----
        -   Does not support automatically setting index column(s) similar
            to how ``to_pandas`` works for PyArrow Tables.

        Examples
        --------
        >>> import cudf
        >>> import pyarrow as pa
        >>> data = pa.table({"a":[1, 2, 3], "b":[4, 5, 6]})
        >>> cudf.DataFrame.from_arrow(data)
           a  b
        0  1  4
        1  2  5
        2  3  6
        """
        index_col = None
        if isinstance(table, pa.Table) and isinstance(
            table.schema.pandas_metadata, dict
        ):
            index_col = table.schema.pandas_metadata["index_columns"]

        out = super().from_arrow(table)

        if index_col:
            if isinstance(index_col[0], dict):
                out = out.set_index(
                    cudf.RangeIndex(
                        index_col[0]["start"],
                        index_col[0]["stop"],
                        name=index_col[0]["name"],
                    )
                )
            else:
                out = out.set_index(index_col[0])

        return out

    def to_arrow(self, preserve_index=True):
        """
        Convert to a PyArrow Table.

        Parameters
        ----------
        preserve_index : bool, default True
            whether index column and its meta data needs to be saved or not

        Returns
        -------
        PyArrow Table

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
        >>> df.to_arrow(preserve_index=False)
        pyarrow.Table
        a: int64
        b: int64
        """

        data = self.copy(deep=False)
        index_descr = []
        if preserve_index:
            if isinstance(self.index, cudf.RangeIndex):
                descr = {
                    "kind": "range",
                    "name": self.index.name,
                    "start": self.index._start,
                    "stop": self.index._stop,
                    "step": 1,
                }
            else:
                if isinstance(self.index, MultiIndex):
                    gen_names = tuple(
                        f"level_{i}"
                        for i, _ in enumerate(self.index._data.names)
                    )
                else:
                    gen_names = (
                        self.index.names
                        if self.index.name is not None
                        else ("index",)
                    )
                for gen_name, col_name in zip(
                    gen_names, self.index._data.names
                ):
                    data.insert(
                        data.shape[1], gen_name, self.index._data[col_name]
                    )
                descr = gen_names[0]
            index_descr.append(descr)

        out = super(DataFrame, data).to_arrow()
        metadata = pa.pandas_compat.construct_metadata(
            columns_to_convert=[self[col] for col in self._data.names],
            df=self,
            column_names=out.schema.names,
            index_levels=[self.index],
            index_descriptors=index_descr,
            preserve_index=preserve_index,
            types=out.schema.types,
        )

        return out.replace_schema_metadata(metadata)

    def to_records(self, index=True):
        """Convert to a numpy recarray

        Parameters
        ----------
        index : bool
            Whether to include the index in the output.

        Returns
        -------
        numpy recarray
        """
        members = [("index", self.index.dtype)] if index else []
        members += [(col, self[col].dtype) for col in self._data.names]
        dtype = np.dtype(members)
        ret = np.recarray(len(self), dtype=dtype)
        if index:
            ret["index"] = self.index.to_numpy()
        for col in self._data.names:
            ret[col] = self[col].to_numpy()
        return ret

    @classmethod
    def from_records(cls, data, index=None, columns=None, nan_as_null=False):
        """
        Convert structured or record ndarray to DataFrame.

        Parameters
        ----------
        data : numpy structured dtype or recarray of ndim=2
        index : str, array-like
            The name of the index column in *data*.
            If None, the default index is used.
        columns : list of str
            List of column names to include.

        Returns
        -------
        DataFrame
        """
        if data.ndim != 1 and data.ndim != 2:
            raise ValueError(
                f"records dimension expected 1 or 2 but found {data.ndim}"
            )

        num_cols = len(data[0])

        if columns is None and data.dtype.names is None:
            names = [i for i in range(num_cols)]

        elif data.dtype.names is not None:
            names = data.dtype.names

        else:
            if len(columns) != num_cols:
                raise ValueError(
                    f"columns length expected {num_cols} "
                    f"but found {len(columns)}"
                )
            names = columns

        df = DataFrame()

        if data.ndim == 2:
            for i, k in enumerate(names):
                df._data[k] = column.as_column(
                    data[:, i], nan_as_null=nan_as_null
                )
        elif data.ndim == 1:
            for k in names:
                df._data[k] = column.as_column(
                    data[k], nan_as_null=nan_as_null
                )

        if index is None:
            df._index = RangeIndex(start=0, stop=len(data))
        elif is_scalar(index):
            df._index = RangeIndex(start=0, stop=len(data))
            df = df.set_index(index)
        else:
            df._index = as_index(index)
        return df

    @classmethod
    def _from_arrays(cls, data, index=None, columns=None, nan_as_null=False):
        """Convert a numpy/cupy array to DataFrame.

        Parameters
        ----------
        data : numpy/cupy array of ndim 1 or 2,
            dimensions greater than 2 are not supported yet.
        index : Index or array-like
            Index to use for resulting frame. Will default to
            RangeIndex if no indexing information part of input data and
            no index provided.
        columns : list of str
            List of column names to include.

        Returns
        -------
        DataFrame
        """

        data = cupy.asarray(data)
        if data.ndim != 1 and data.ndim != 2:
            raise ValueError(
                f"records dimension expected 1 or 2 but found: {data.ndim}"
            )

        if data.ndim == 2:
            num_cols = data.shape[1]
        else:
            # Since we validate ndim to be either 1 or 2 above,
            # this case can be assumed to be ndim == 1.
            num_cols = 1

        if columns is None:
            names = [i for i in range(num_cols)]
        else:
            if len(columns) != num_cols:
                raise ValueError(
                    f"columns length expected {num_cols} but "
                    f"found {len(columns)}"
                )
            names = columns

        df = cls()
        if data.ndim == 2:
            for i, k in enumerate(names):
                df._data[k] = column.as_column(
                    data[:, i], nan_as_null=nan_as_null
                )
        elif data.ndim == 1:
            df._data[names[0]] = column.as_column(
                data, nan_as_null=nan_as_null
            )

        if index is None:
            df._index = RangeIndex(start=0, stop=len(data))
        else:
            df._index = as_index(index)
        return df

    def interpolate(
        self,
        method="linear",
        axis=0,
        limit=None,
        inplace=False,
        limit_direction=None,
        limit_area=None,
        downcast=None,
        **kwargs,
    ):

        if all(dt == np.dtype("object") for dt in self.dtypes):
            raise TypeError(
                "Cannot interpolate with all object-dtype "
                "columns in the DataFrame. Try setting at "
                "least one column to a numeric dtype."
            )

        return super().interpolate(
            method=method,
            axis=axis,
            limit=limit,
            inplace=inplace,
            limit_direction=limit_direction,
            limit_area=limit_area,
            downcast=downcast,
            **kwargs,
        )

    def quantile(
        self,
        q=0.5,
        axis=0,
        numeric_only=True,
        interpolation="linear",
        columns=None,
        exact=True,
    ):
        """
        Return values at the given quantile.

        Parameters
        ----------

        q : float or array-like
            0 <= q <= 1, the quantile(s) to compute
        axis : int
            axis is a NON-FUNCTIONAL parameter
        numeric_only : bool, default True
            If False, the quantile of datetime and timedelta data will be
            computed as well.
        interpolation : {`linear`, `lower`, `higher`, `midpoint`, `nearest`}
            This parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points i and j.
            Default ``linear``.
        columns : list of str
            List of column names to include.
        exact : boolean
            Whether to use approximate or exact quantile algorithm.

        Returns
        -------
        Series or DataFrame
            If q is an array or numeric_only is set to False, a DataFrame
            will be returned where index is q, the columns are the columns
            of self, and the values are the quantile.

            If q is a float, a Series will be returned where the index is
            the columns of self and the values are the quantiles.

        Notes
        -----
        One notable difference from Pandas is when DataFrame is of
        non-numeric types and result is expected to be a Series in case of
        Pandas. cuDF will return a DataFrame as it doesn't support mixed
        types under Series.

        Examples
        --------
        >>> import cupy as cp
        >>> import cudf
        >>> df = cudf.DataFrame(cp.array([[1, 1], [2, 10], [3, 100], [4, 100]]),
        ...                   columns=['a', 'b'])
        >>> df
           a    b
        0  1    1
        1  2   10
        2  3  100
        3  4  100
        >>> df.quantile(0.1)
        a    1.3
        b    3.7
        Name: 0.1, dtype: float64
        >>> df.quantile([.1, .5])
               a     b
        0.1  1.3   3.7
        0.5  2.5  55.0
        """  # noqa: E501
        if axis not in (0, None):
            raise NotImplementedError("axis is not implemented yet")

        if numeric_only:
            data_df = self.select_dtypes(
                include=[np.number], exclude=["datetime64", "timedelta64"]
            )
        else:
            data_df = self

        if columns is None:
            columns = data_df._data.names

        result = DataFrame()

        for k in data_df._data.names:

            if k in columns:
                res = data_df[k].quantile(
                    q,
                    interpolation=interpolation,
                    exact=exact,
                    quant_index=False,
                )
                if (
                    not isinstance(
                        res, (numbers.Number, pd.Timestamp, pd.Timedelta)
                    )
                    and len(res) == 0
                ):
                    res = column.column_empty_like(
                        q, dtype=data_df[k].dtype, masked=True, newsize=len(q)
                    )
                result[k] = column.as_column(res)

        if isinstance(q, numbers.Number) and numeric_only:
            result = result.fillna(np.nan)
            result = result.iloc[0]
            result.index = as_index(data_df.columns)
            result.name = q
            return result
        else:
            q = list(map(float, [q] if isinstance(q, numbers.Number) else q))
            result.index = q
            return result

    def quantiles(self, q=0.5, interpolation="nearest"):
        """
        Return values at the given quantile.

        Parameters
        ----------

        q : float or array-like
            0 <= q <= 1, the quantile(s) to compute
        interpolation : {`lower`, `higher`, `nearest`}
            This parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points i and j.
            Default 'nearest'.

        Returns
        -------

        DataFrame
        """
        if isinstance(q, numbers.Number):
            q_is_number = True
            q = [float(q)]
        elif pd.api.types.is_list_like(q):
            q_is_number = False
        else:
            msg = "`q` must be either a single element or list"
            raise TypeError(msg)

        result = self._quantiles(q, interpolation.upper())

        if q_is_number:
            result = result.transpose()
            return Series(
                data=result._columns[0], index=result.index, name=q[0]
            )
        else:
            result.index = as_index(q)
            return result

    def isin(self, values):
        """
        Whether each element in the DataFrame is contained in values.

        Parameters
        ----------

        values : iterable, Series, DataFrame or dict
            The result will only be true at a location if all
            the labels match. If values is a Series, that’s the index.
            If values is a dict, the keys must be the column names,
            which must match. If values is a DataFrame, then both the
            index and column labels must match.

        Returns
        -------
        DataFrame:
            DataFrame of booleans showing whether each element in
            the DataFrame is contained in values.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'num_legs': [2, 4], 'num_wings': [2, 0]},
        ...                     index=['falcon', 'dog'])
        >>> df
                num_legs  num_wings
        falcon         2          2
        dog            4          0

        When ``values`` is a list check whether every value in the DataFrame
        is present in the list (which animals have 0 or 2 legs or wings)

        >>> df.isin([0, 2])
                num_legs  num_wings
        falcon      True       True
        dog        False       True

        When ``values`` is a dict, we can pass values to check for each
        column separately:

        >>> df.isin({'num_wings': [0, 3]})
                num_legs  num_wings
        falcon     False      False
        dog        False       True

        When ``values`` is a Series or DataFrame the index and column must
        match. Note that 'falcon' does not match based on the number of legs
        in other.

        >>> other = cudf.DataFrame({'num_legs': [8, 2], 'num_wings': [0, 2]},
        ...                         index=['spider', 'falcon'])
        >>> df.isin(other)
                num_legs  num_wings
        falcon      True       True
        dog        False      False
        """
        if isinstance(values, dict):

            result_df = DataFrame()

            for col in self._data.names:
                if col in values:
                    val = values[col]
                    result_df[col] = self._data[col].isin(val)
                else:
                    result_df[col] = column.full(
                        size=len(self), fill_value=False, dtype="bool"
                    )

            result_df.index = self.index
            return result_df
        elif isinstance(values, Series):
            values = values.reindex(self.index)

            result = DataFrame()
            # TODO: propagate nulls through isin
            # https://github.com/rapidsai/cudf/issues/7556
            for col in self._data.names:
                if isinstance(
                    self[col]._column, cudf.core.column.CategoricalColumn
                ) and isinstance(
                    values._column, cudf.core.column.CategoricalColumn
                ):
                    res = (self._data[col] == values._column).fillna(False)
                    result[col] = res
                elif (
                    isinstance(
                        self[col]._column, cudf.core.column.CategoricalColumn
                    )
                    or np.issubdtype(self[col].dtype, cudf.dtype("object"))
                ) or (
                    isinstance(
                        values._column, cudf.core.column.CategoricalColumn
                    )
                    or np.issubdtype(values.dtype, cudf.dtype("object"))
                ):
                    result[col] = utils.scalar_broadcast_to(False, len(self))
                else:
                    result[col] = (self._data[col] == values._column).fillna(
                        False
                    )

            result.index = self.index
            return result
        elif isinstance(values, DataFrame):
            values = values.reindex(self.index)

            result = DataFrame()
            for col in self._data.names:
                if col in values.columns:
                    result[col] = (
                        self._data[col] == values[col]._column
                    ).fillna(False)
                else:
                    result[col] = utils.scalar_broadcast_to(False, len(self))
            result.index = self.index
            return result
        else:
            if not is_list_like(values):
                raise TypeError(
                    f"only list-like or dict-like objects are "
                    f"allowed to be passed to DataFrame.isin(), "
                    f"you passed a "
                    f"'{type(values).__name__}'"
                )

            result_df = DataFrame()

            for col in self._data.names:
                result_df[col] = self._data[col].isin(values)
            result_df.index = self.index
            return result_df

    #
    # Stats
    #
    def _prepare_for_rowwise_op(self, method, skipna):
        """Prepare a DataFrame for CuPy-based row-wise operations."""

        if method not in _cupy_nan_methods_map and any(
            col.nullable for col in self._columns
        ):
            msg = (
                f"Row-wise operations to calculate '{method}' do not "
                f"currently support columns with null values. "
                f"Consider removing them with .dropna() "
                f"or using .fillna()."
            )
            raise ValueError(msg)

        is_pure_dt = all(is_datetime_dtype(dt) for dt in self.dtypes)

        if not is_pure_dt:
            filtered = self.select_dtypes(include=[np.number, np.bool_])
        else:
            filtered = self.copy(deep=False)

        common_dtype = find_common_type(filtered.dtypes)

        if filtered._num_columns < self._num_columns:
            msg = (
                "Row-wise operations currently only support int, float "
                "and bool dtypes. Non numeric columns are ignored."
            )
            warnings.warn(msg)

        if not skipna and any(col.nullable for col in filtered._columns):
            mask = DataFrame(
                {
                    name: filtered._data[name]._get_mask_as_column()
                    if filtered._data[name].nullable
                    else column.full(len(filtered._data[name]), True)
                    for name in filtered._data.names
                }
            )
            mask = mask.all(axis=1)
        else:
            mask = None

        coerced = filtered.astype(common_dtype, copy=False)
        if is_pure_dt:
            # Further convert into cupy friendly types
            coerced = coerced.astype("int64", copy=False)
        return coerced, mask, common_dtype

    def count(self, axis=0, level=None, numeric_only=False, **kwargs):
        """
        Count ``non-NA`` cells for each column or row.

        The values ``None``, ``NaN``, ``NaT`` are considered ``NA``.

        Returns
        -------
        Series
            For each column/row the number of non-NA/null entries.

        Notes
        -----
        Parameters currently not supported are `axis`, `level`, `numeric_only`.

        Examples
        --------
        >>> import cudf
        >>> import numpy as np
        >>> df = cudf.DataFrame({"Person":
        ...        ["John", "Myla", "Lewis", "John", "Myla"],
        ...        "Age": [24., np.nan, 21., 33, 26],
        ...        "Single": [False, True, True, True, False]})
        >>> df.count()
        Person    5
        Age       4
        Single    5
        dtype: int64
        """
        axis = self._get_axis_from_axis_arg(axis)
        if axis != 0:
            raise NotImplementedError("Only axis=0 is currently supported.")

        return Series._from_data(
            {None: [self._data[col].valid_count for col in self._data.names]},
            as_index(self._data.names),
        )

    _SUPPORT_AXIS_LOOKUP = {
        0: 0,
        1: 1,
        None: 0,
        "index": 0,
        "columns": 1,
    }

    def _reduce(
        self, op, axis=None, level=None, numeric_only=None, **kwargs,
    ):
        if level is not None:
            raise NotImplementedError("level parameter is not implemented yet")

        if numeric_only not in (None, True):
            raise NotImplementedError(
                "numeric_only parameter is not implemented yet"
            )
        axis = self._get_axis_from_axis_arg(axis)

        if axis == 0:
            result = [
                getattr(self._data[col], op)(**kwargs)
                for col in self._data.names
            ]

            return Series._from_data(
                {None: result}, as_index(self._data.names)
            )
        elif axis == 1:
            return self._apply_cupy_method_axis_1(op, **kwargs)

    def _scan(
        self, op, axis=None, *args, **kwargs,
    ):
        axis = self._get_axis_from_axis_arg(axis)

        if axis == 0:
            return super()._scan(op, axis=axis, *args, **kwargs)
        elif axis == 1:
            return self._apply_cupy_method_axis_1(f"cum{op}", **kwargs)

    def mode(self, axis=0, numeric_only=False, dropna=True):
        """
        Get the mode(s) of each element along the selected axis.

        The mode of a set of values is the value that appears most often.
        It can be multiple values.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to iterate over while searching for the mode:

            - 0 or 'index' : get mode of each column
            - 1 or 'columns' : get mode of each row.
        numeric_only : bool, default False
            If True, only apply to numeric columns.
        dropna : bool, default True
            Don't consider counts of NA/NaN/NaT.

        Returns
        -------
        DataFrame
            The modes of each column or row.

        See Also
        --------
        cudf.Series.mode : Return the highest frequency value
            in a Series.
        cudf.Series.value_counts : Return the counts of values
            in a Series.

        Notes
        -----
        ``axis`` parameter is currently not supported.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({
        ...     "species": ["bird", "mammal", "arthropod", "bird"],
        ...     "legs": [2, 4, 8, 2],
        ...     "wings": [2.0, None, 0.0, None]
        ... })
        >>> df
             species  legs wings
        0       bird     2   2.0
        1     mammal     4  <NA>
        2  arthropod     8   0.0
        3       bird     2  <NA>

        By default, missing values are not considered, and the mode of wings
        are both 0 and 2. The second row of species and legs contains ``NA``,
        because they have only one mode, but the DataFrame has two rows.

        >>> df.mode()
          species  legs  wings
        0    bird     2    0.0
        1    <NA>  <NA>    2.0

        Setting ``dropna=False``, ``NA`` values are considered and they can be
        the mode (like for wings).

        >>> df.mode(dropna=False)
          species  legs wings
        0    bird     2  <NA>

        Setting ``numeric_only=True``, only the mode of numeric columns is
        computed, and columns of other types are ignored.

        >>> df.mode(numeric_only=True)
           legs  wings
        0     2    0.0
        1  <NA>    2.0
        """
        if axis not in (0, "index"):
            raise NotImplementedError("Only axis=0 is currently supported")

        if numeric_only:
            data_df = self.select_dtypes(
                include=[np.number], exclude=["datetime64", "timedelta64"]
            )
        else:
            data_df = self

        mode_results = [
            data_df[col].mode(dropna=dropna) for col in data_df._data
        ]

        if len(mode_results) == 0:
            return DataFrame()

        df = cudf.concat(mode_results, axis=1)
        if isinstance(df, Series):
            df = df.to_frame()

        df.columns = data_df.columns

        return df

    def kurtosis(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs
    ):
        obj = self.select_dtypes(include=[np.number, np.bool_])
        return super(DataFrame, obj).kurtosis(
            axis, skipna, level, numeric_only, **kwargs
        )

    def skew(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs
    ):
        obj = self.select_dtypes(include=[np.number, np.bool_])
        return super(DataFrame, obj).skew(
            axis, skipna, level, numeric_only, **kwargs
        )

    def all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        obj = self.select_dtypes(include="bool") if bool_only else self
        return super(DataFrame, obj).all(axis, skipna, level, **kwargs)

    def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        obj = self.select_dtypes(include="bool") if bool_only else self
        return super(DataFrame, obj).any(axis, skipna, level, **kwargs)

    def _apply_cupy_method_axis_1(self, method, *args, **kwargs):
        # This method uses cupy to perform scans and reductions along rows of a
        # DataFrame. Since cuDF is designed around columnar storage and
        # operations, we convert DataFrames to 2D cupy arrays for these ops.

        # for dask metadata compatibility
        skipna = kwargs.pop("skipna", None)
        skipna = True if skipna is None else skipna
        if method not in _cupy_nan_methods_map and skipna not in (
            None,
            True,
            1,
        ):
            raise NotImplementedError(
                f"Row-wise operations to calculate '{method}'"
                f" currently do not support `skipna=False`."
            )

        level = kwargs.pop("level", None)
        if level not in (None,):
            raise NotImplementedError(
                "Row-wise operations currently do not support `level`."
            )

        numeric_only = kwargs.pop("numeric_only", None)
        if numeric_only not in (None, True):
            raise NotImplementedError(
                "Row-wise operations currently do not "
                "support `numeric_only=False`."
            )

        min_count = kwargs.pop("min_count", None)
        if min_count not in (None, 0):
            raise NotImplementedError(
                "Row-wise operations currently do not support `min_count`."
            )

        bool_only = kwargs.pop("bool_only", None)
        if bool_only not in (None, True):
            raise NotImplementedError(
                "Row-wise operations currently do not support `bool_only`."
            )

        # This parameter is only necessary for axis 0 reductions that cuDF
        # performs internally. cupy already upcasts smaller integer/bool types
        # to int64 when accumulating.
        kwargs.pop("cast_to_int", None)

        prepared, mask, common_dtype = self._prepare_for_rowwise_op(
            method, skipna
        )
        for col in prepared._data.names:
            if prepared._data[col].nullable:
                prepared._data[col] = (
                    prepared._data[col]
                    .astype(
                        cudf.utils.dtypes.get_min_float_dtype(
                            prepared._data[col]
                        )
                        if not is_datetime_dtype(common_dtype)
                        else cudf.dtype("float64")
                    )
                    .fillna(np.nan)
                )
        arr = cupy.asarray(prepared.as_gpu_matrix())

        if skipna is not False and method in _cupy_nan_methods_map:
            method = _cupy_nan_methods_map[method]

        result = getattr(cupy, method)(arr, axis=1, **kwargs)

        if result.ndim == 1:
            type_coerced_methods = {
                "count",
                "min",
                "max",
                "sum",
                "prod",
                "cummin",
                "cummax",
                "cumsum",
                "cumprod",
            }
            result_dtype = (
                common_dtype
                if method in type_coerced_methods
                or is_datetime_dtype(common_dtype)
                else None
            )
            result = column.as_column(result, dtype=result_dtype)
            if mask is not None:
                result = result.set_mask(
                    cudf._lib.transform.bools_to_mask(mask._column)
                )
            return Series(result, index=self.index, dtype=result_dtype,)
        else:
            result_df = DataFrame(result).set_index(self.index)
            result_df.columns = prepared.columns
            return result_df

    def _columns_view(self, columns):
        """
        Return a subset of the DataFrame's columns as a view.
        """
        return DataFrame(
            {col: self._data[col] for col in columns}, index=self.index
        )

    def select_dtypes(self, include=None, exclude=None):
        """Return a subset of the DataFrame’s columns based on the column dtypes.

        Parameters
        ----------
        include : str or list
            which columns to include based on dtypes
        exclude : str or list
            which columns to exclude based on dtypes

        Returns
        -------
        DataFrame
            The subset of the frame including the dtypes
            in ``include`` and excluding the dtypes in ``exclude``.

        Raises
        ------
        ValueError
            - If both of ``include`` and ``exclude`` are empty
            - If ``include`` and ``exclude`` have overlapping elements

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2] * 3,
        ...                    'b': [True, False] * 3,
        ...                    'c': [1.0, 2.0] * 3})
        >>> df
           a      b    c
        0  1   True  1.0
        1  2  False  2.0
        2  1   True  1.0
        3  2  False  2.0
        4  1   True  1.0
        5  2  False  2.0
        >>> df.select_dtypes(include='bool')
               b
        0   True
        1  False
        2   True
        3  False
        4   True
        5  False
        >>> df.select_dtypes(include=['float64'])
             c
        0  1.0
        1  2.0
        2  1.0
        3  2.0
        4  1.0
        5  2.0
        >>> df.select_dtypes(exclude=['int'])
               b    c
        0   True  1.0
        1  False  2.0
        2   True  1.0
        3  False  2.0
        4   True  1.0
        5  False  2.0
        """

        # code modified from:
        # https://github.com/pandas-dev/pandas/blob/master/pandas/core/frame.py#L3196

        if not isinstance(include, (list, tuple)):
            include = (include,) if include is not None else ()
        if not isinstance(exclude, (list, tuple)):
            exclude = (exclude,) if exclude is not None else ()

        df = DataFrame(index=self.index)

        # cudf_dtype_from_pydata_dtype can distinguish between
        # np.float and np.number
        selection = tuple(map(frozenset, (include, exclude)))

        if not any(selection):
            raise ValueError(
                "at least one of include or exclude must be nonempty"
            )

        include, exclude = map(
            lambda x: frozenset(map(cudf_dtype_from_pydata_dtype, x)),
            selection,
        )

        # can't both include AND exclude!
        if not include.isdisjoint(exclude):
            raise ValueError(
                f"include and exclude overlap on {(include & exclude)}"
            )

        # include all subtypes
        include_subtypes = set()
        for dtype in self.dtypes:
            for i_dtype in include:
                # category handling
                if is_categorical_dtype(i_dtype):
                    include_subtypes.add(i_dtype)
                elif inspect.isclass(dtype.type):
                    if issubclass(dtype.type, i_dtype):
                        include_subtypes.add(dtype.type)

        # exclude all subtypes
        exclude_subtypes = set()
        for dtype in self.dtypes:
            for e_dtype in exclude:
                # category handling
                if is_categorical_dtype(e_dtype):
                    exclude_subtypes.add(e_dtype)
                elif inspect.isclass(dtype.type):
                    if issubclass(dtype.type, e_dtype):
                        exclude_subtypes.add(dtype.type)

        include_all = set(
            [cudf_dtype_from_pydata_dtype(d) for d in self.dtypes]
        )

        if include:
            inclusion = include_all & include_subtypes
        elif exclude:
            inclusion = include_all
        else:
            inclusion = set()
        # remove all exclude types
        inclusion = inclusion - exclude_subtypes

        for k, col in self._data.items():
            infered_type = cudf_dtype_from_pydata_dtype(col.dtype)
            if infered_type in inclusion:
                df.insert(len(df._data), k, col)

        return df

    @ioutils.doc_to_parquet()
    def to_parquet(self, path, *args, **kwargs):
        """{docstring}"""
        from cudf.io import parquet as pq

        return pq.to_parquet(self, path, *args, **kwargs)

    @ioutils.doc_to_feather()
    def to_feather(self, path, *args, **kwargs):
        """{docstring}"""
        from cudf.io import feather as feather

        feather.to_feather(self, path, *args, **kwargs)

    @ioutils.doc_dataframe_to_csv()
    def to_csv(
        self,
        path_or_buf=None,
        sep=",",
        na_rep="",
        columns=None,
        header=True,
        index=True,
        line_terminator="\n",
        chunksize=None,
        encoding=None,
        compression=None,
        **kwargs,
    ):
        """{docstring}"""
        from cudf.io import csv as csv

        return csv.to_csv(
            self,
            path_or_buf=path_or_buf,
            sep=sep,
            na_rep=na_rep,
            columns=columns,
            header=header,
            index=index,
            line_terminator=line_terminator,
            chunksize=chunksize,
            encoding=encoding,
            compression=compression,
            **kwargs,
        )

    @ioutils.doc_to_orc()
    def to_orc(self, fname, compression=None, *args, **kwargs):
        """{docstring}"""
        from cudf.io import orc as orc

        orc.to_orc(self, fname, compression, *args, **kwargs)

    def stack(self, level=-1, dropna=True):
        """Stack the prescribed level(s) from columns to index

        Return a reshaped Series

        Parameters
        ----------
        dropna : bool, default True
            Whether to drop rows in the resulting Series with missing values.

        Returns
        -------
        The stacked cudf.Series

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a':[0,1,3], 'b':[1,2,4]})
        >>> df.stack()
        0  a    0
           b    1
        1  a    1
           b    2
        2  a    3
           b    4
        dtype: int64
        """
        assert level in (None, -1)
        repeated_index = self.index.repeat(self.shape[1])
        name_index = Frame({0: self._column_names}).tile(self.shape[0])
        new_index = list(repeated_index._columns) + [name_index._columns[0]]
        if isinstance(self._index, MultiIndex):
            index_names = self._index.names + [None]
        else:
            index_names = [None] * len(new_index)
        new_index = MultiIndex.from_frame(
            DataFrame(dict(zip(range(0, len(new_index)), new_index))),
            names=index_names,
        )

        # Collect datatypes and cast columns as that type
        common_type = np.result_type(*self.dtypes)
        homogenized = DataFrame(
            {
                c: (
                    self._data[c].astype(common_type)
                    if not np.issubdtype(self._data[c].dtype, common_type)
                    else self._data[c]
                )
                for c in self._data
            }
        )

        data_col = libcudf.reshape.interleave_columns(homogenized)

        result = Series(data=data_col, index=new_index)
        if dropna:
            return result.dropna()
        else:
            return result

    def cov(self, **kwargs):
        """Compute the covariance matrix of a DataFrame.

        Parameters
        ----------
        **kwargs
            Keyword arguments to be passed to cupy.cov

        Returns
        -------
        cov : DataFrame
        """
        cov = cupy.cov(self.values, rowvar=False)
        df = DataFrame(cupy.asfortranarray(cov)).set_index(self.columns)
        df.columns = self.columns
        return df

    def corr(self):
        """Compute the correlation matrix of a DataFrame."""
        corr = cupy.corrcoef(self.values, rowvar=False)
        df = DataFrame(cupy.asfortranarray(corr)).set_index(self.columns)
        df.columns = self.columns
        return df

    def to_struct(self, name=None):
        """
        Return a struct Series composed of the columns of the DataFrame.

        Parameters
        ----------
        name: optional
            Name of the resulting Series

        Notes
        -----
        Note that a copy of the columns is made.
        """
        col = cudf.core.column.build_struct_column(
            names=self._data.names, children=self._data.columns, size=len(self)
        )
        return cudf.Series._from_data(
            cudf.core.column_accessor.ColumnAccessor(
                {name: col.copy(deep=True)}
            ),
            index=self.index,
            name=name,
        )

    def keys(self):
        """
        Get the columns.
        This is index for Series, columns for DataFrame.

        Returns
        -------
        Index
            Columns of DataFrame.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'one' : [1, 2, 3], 'five' : ['a', 'b', 'c']})
        >>> df
           one five
        0    1    a
        1    2    b
        2    3    c
        >>> df.keys()
        Index(['one', 'five'], dtype='object')
        >>> df = cudf.DataFrame(columns=[0, 1, 2, 3])
        >>> df
        Empty DataFrame
        Columns: [0, 1, 2, 3]
        Index: []
        >>> df.keys()
        Int64Index([0, 1, 2, 3], dtype='int64')
        """
        return self.columns

    def itertuples(self, index=True, name="Pandas"):
        raise TypeError(
            "cuDF does not support iteration of DataFrame "
            "via itertuples. Consider using "
            "`.to_pandas().itertuples()` "
            "if you wish to iterate over namedtuples."
        )

    def iterrows(self):
        raise TypeError(
            "cuDF does not support iteration of DataFrame "
            "via iterrows. Consider using "
            "`.to_pandas().iterrows()` "
            "if you wish to iterate over each row."
        )

    def append(
        self, other, ignore_index=False, verify_integrity=False, sort=False
    ):
        """
        Append rows of `other` to the end of caller, returning a new object.
        Columns in `other` that are not in the caller are added as new columns.

        Parameters
        ----------
        other : DataFrame or Series/dict-like object, or list of these
            The data to append.
        ignore_index : bool, default False
            If True, do not use the index labels.
        sort : bool, default False
            Sort columns ordering if the columns of
            `self` and `other` are not aligned.
        verify_integrity : bool, default False
            This Parameter is currently not supported.

        Returns
        -------
        DataFrame

        See Also
        --------
        cudf.concat : General function to concatenate DataFrame or
            objects.

        Notes
        -----
        If a list of dict/series is passed and the keys are all contained in
        the DataFrame's index, the order of the columns in the resulting
        DataFrame will be unchanged.
        Iteratively appending rows to a cudf DataFrame can be more
        computationally intensive than a single concatenate. A better
        solution is to append those rows to a list and then concatenate
        the list with the original DataFrame all at once.
        `verify_integrity` parameter is not supported yet.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame([[1, 2], [3, 4]], columns=list('AB'))
        >>> df
           A  B
        0  1  2
        1  3  4
        >>> df2 = cudf.DataFrame([[5, 6], [7, 8]], columns=list('AB'))
        >>> df2
           A  B
        0  5  6
        1  7  8
        >>> df.append(df2)
           A  B
        0  1  2
        1  3  4
        0  5  6
        1  7  8

        With `ignore_index` set to True:

        >>> df.append(df2, ignore_index=True)
           A  B
        0  1  2
        1  3  4
        2  5  6
        3  7  8

        The following, while not recommended methods for generating DataFrames,
        show two ways to generate a DataFrame from multiple data sources.
        Less efficient:

        >>> df = cudf.DataFrame(columns=['A'])
        >>> for i in range(5):
        ...     df = df.append({'A': i}, ignore_index=True)
        >>> df
           A
        0  0
        1  1
        2  2
        3  3
        4  4

        More efficient than above:

        >>> cudf.concat([cudf.DataFrame([i], columns=['A']) for i in range(5)],
        ...           ignore_index=True)
           A
        0  0
        1  1
        2  2
        3  3
        4  4
        """
        if verify_integrity not in (None, False):
            raise NotImplementedError(
                "verify_integrity parameter is not supported yet."
            )

        if isinstance(other, dict):
            if not ignore_index:
                raise TypeError("Can only append a dict if ignore_index=True")
            other = DataFrame(other)
            result = cudf.concat(
                [self, other], ignore_index=ignore_index, sort=sort
            )
            return result
        elif isinstance(other, Series):
            if other.name is None and not ignore_index:
                raise TypeError(
                    "Can only append a Series if ignore_index=True "
                    "or if the Series has a name"
                )

            current_cols = self.columns
            combined_columns = other.index.to_pandas()
            if len(current_cols):

                if cudf.utils.dtypes.is_mixed_with_object_dtype(
                    current_cols, combined_columns
                ):
                    raise TypeError(
                        "cudf does not support mixed types, please type-cast "
                        "the column index of dataframe and index of series "
                        "to same dtypes."
                    )

                combined_columns = current_cols.union(
                    combined_columns, sort=False
                )

            if sort:
                combined_columns = combined_columns.sort_values()

            other = other.reindex(combined_columns, copy=False).to_frame().T
            if not current_cols.equals(combined_columns):
                self = self.reindex(columns=combined_columns)
        elif isinstance(other, list):
            if not other:
                pass
            elif not isinstance(other[0], DataFrame):
                other = DataFrame(other)
                if (self.columns.get_indexer(other.columns) >= 0).all():
                    other = other.reindex(columns=self.columns)

        if is_list_like(other):
            to_concat = [self, *other]
        else:
            to_concat = [self, other]

        return cudf.concat(to_concat, ignore_index=ignore_index, sort=sort)

    @copy_docstring(reshape.pivot)
    def pivot(self, index, columns, values=None):

        return cudf.core.reshape.pivot(
            self, index=index, columns=columns, values=values
        )

    @copy_docstring(reshape.unstack)
    def unstack(self, level=-1, fill_value=None):
        return cudf.core.reshape.unstack(
            self, level=level, fill_value=fill_value
        )

    def explode(self, column, ignore_index=False):
        """
        Transform each element of a list-like to a row, replicating index
        values.

        Parameters
        ----------
        column : str or tuple
            Column to explode.
        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> import cudf
        >>> cudf.DataFrame(
                {"a": [[1, 2, 3], [], None, [4, 5]], "b": [11, 22, 33, 44]})
                   a   b
        0  [1, 2, 3]  11
        1         []  22
        2       None  33
        3     [4, 5]  44
        >>> df.explode('a')
              a   b
        0     1  11
        0     2  11
        0     3  11
        1  <NA>  22
        2  <NA>  33
        3     4  44
        3     5  44
        """
        if column not in self._column_names:
            raise KeyError(column)

        if not is_list_dtype(self._data[column].dtype):
            data = self._data.copy(deep=True)
            idx = None if ignore_index else self._index.copy(deep=True)
            return self.__class__._from_data(data, index=idx)

        return super()._explode(column, ignore_index)

    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ):
        return df_protocol.__dataframe__(
            self, nan_as_null=nan_as_null, allow_copy=allow_copy
        )


def from_dataframe(df, allow_copy=False):
    return df_protocol.from_dataframe(df, allow_copy=allow_copy)


def make_binop_func(op, postprocess=None):
    # This function is used to wrap binary operations in Frame with an
    # appropriate API for DataFrame as required for pandas compatibility. The
    # main effect is reordering and error-checking parameters in
    # DataFrame-specific ways. The postprocess argument is a callable that may
    # optionally be provided to modify the result of the binop if additional
    # processing is needed for pandas compatibility. The callable must have the
    # signature
    # def postprocess(left, right, output)
    # where left and right are the inputs to the binop and output is the result
    # of calling the wrapped Frame binop.
    wrapped_func = getattr(Frame, op)

    @functools.wraps(wrapped_func)
    def wrapper(self, other, axis="columns", level=None, fill_value=None):
        if axis not in (1, "columns"):
            raise NotImplementedError("Only axis=1 supported at this time.")
        output = wrapped_func(self, other, axis, level, fill_value)
        if postprocess is None:
            return output
        return postprocess(self, other, output)

    # functools.wraps copies module level attributes to `wrapper` and sets
    # __wrapped__ attributes to `wrapped_func`. Cpython looks up the signature
    # string of a function by recursively delving into __wrapped__ until
    # it hits the first function that has __signature__ attribute set. To make
    # the signature string of `wrapper` matches with its actual parameter list,
    # we directly set the __signature__ attribute of `wrapper` below.

    new_sig = inspect.signature(
        lambda self, other, axis="columns", level=None, fill_value=None: None
    )

    wrapper.__signature__ = new_sig
    return wrapper


# Wrap arithmetic Frame binop functions with the expected API for Series.
for binop in [
    "add",
    "radd",
    "subtract",
    "sub",
    "rsub",
    "multiply",
    "mul",
    "rmul",
    "mod",
    "rmod",
    "pow",
    "rpow",
    "floordiv",
    "rfloordiv",
    "truediv",
    "div",
    "divide",
    "rtruediv",
    "rdiv",
]:
    setattr(DataFrame, binop, make_binop_func(binop))


def _make_replacement_func(value):
    # This function generates a postprocessing function suitable for use with
    # make_binop_func that fills null columns with the desired fill value.

    def func(left, right, output):
        # This function may be passed as the postprocess argument to
        # make_binop_func. Columns that are only present in one of the inputs
        # will be null in the output. This function postprocesses the output to
        # replace those nulls with some desired output.
        if isinstance(right, Series):
            uncommon_columns = set(left._column_names) ^ set(right.index)
        elif isinstance(right, DataFrame):
            uncommon_columns = set(left._column_names) ^ set(
                right._column_names
            )
        elif _is_scalar_or_zero_d_array(right):
            for name, col in output._data.items():
                output._data[name] = col.fillna(value)
            return output
        else:
            return output

        for name in uncommon_columns:
            output._data[name] = column.full(
                size=len(output), fill_value=value, dtype="bool"
            )
        return output

    return func


# The ne comparator needs special postprocessing because elements that missing
# in one operand should be treated as null and result in True in the output
# rather than simply propagating nulls.
DataFrame.ne = make_binop_func("ne", _make_replacement_func(True))


# All other comparison operators needs return False when one of the operands is
# missing in the input.
for binop in [
    "eq",
    "lt",
    "le",
    "gt",
    "ge",
]:
    setattr(
        DataFrame, binop, make_binop_func(binop, _make_replacement_func(False))
    )


def from_pandas(obj, nan_as_null=None):
    """
    Convert certain Pandas objects into the cudf equivalent.

    Supports DataFrame, Series, Index, or MultiIndex.

    Returns
    -------
    DataFrame/Series/Index/MultiIndex
        Return type depends on the passed input.

    Raises
    ------
    TypeError for invalid input type.

    Examples
    --------
    >>> import cudf
    >>> import pandas as pd
    >>> data = [[0, 1], [1, 2], [3, 4]]
    >>> pdf = pd.DataFrame(data, columns=['a', 'b'], dtype=int)
    >>> pdf
       a  b
    0  0  1
    1  1  2
    2  3  4
    >>> gdf = cudf.from_pandas(pdf)
    >>> gdf
       a  b
    0  0  1
    1  1  2
    2  3  4
    >>> type(gdf)
    <class 'cudf.core.dataframe.DataFrame'>
    >>> type(pdf)
    <class 'pandas.core.frame.DataFrame'>

    Converting a Pandas Series to cuDF Series:

    >>> psr = pd.Series(['a', 'b', 'c', 'd'], name='apple')
    >>> psr
    0    a
    1    b
    2    c
    3    d
    Name: apple, dtype: object
    >>> gsr = cudf.from_pandas(psr)
    >>> gsr
    0    a
    1    b
    2    c
    3    d
    Name: apple, dtype: object
    >>> type(gsr)
    <class 'cudf.core.series.Series'>
    >>> type(psr)
    <class 'pandas.core.series.Series'>

    Converting a Pandas Index to cuDF Index:

    >>> pidx = pd.Index([1, 2, 10, 20])
    >>> pidx
    Int64Index([1, 2, 10, 20], dtype='int64')
    >>> gidx = cudf.from_pandas(pidx)
    >>> gidx
    Int64Index([1, 2, 10, 20], dtype='int64')
    >>> type(gidx)
    <class 'cudf.core.index.Int64Index'>
    >>> type(pidx)
    <class 'pandas.core.indexes.numeric.Int64Index'>

    Converting a Pandas MultiIndex to cuDF MultiIndex:

    >>> pmidx = pd.MultiIndex(
    ...         levels=[[1, 3, 4, 5], [1, 2, 5]],
    ...         codes=[[0, 0, 1, 2, 3], [0, 2, 1, 1, 0]],
    ...         names=["x", "y"],
    ...     )
    >>> pmidx
    MultiIndex([(1, 1),
                (1, 5),
                (3, 2),
                (4, 2),
                (5, 1)],
               names=['x', 'y'])
    >>> gmidx = cudf.from_pandas(pmidx)
    >>> gmidx
    MultiIndex([(1, 1),
                (1, 5),
                (3, 2),
                (4, 2),
                (5, 1)],
               names=['x', 'y'])
    >>> type(gmidx)
    <class 'cudf.core.multiindex.MultiIndex'>
    >>> type(pmidx)
    <class 'pandas.core.indexes.multi.MultiIndex'>
    """
    if isinstance(obj, pd.DataFrame):
        return DataFrame.from_pandas(obj, nan_as_null=nan_as_null)
    elif isinstance(obj, pd.Series):
        return Series.from_pandas(obj, nan_as_null=nan_as_null)
    elif isinstance(obj, pd.MultiIndex):
        return MultiIndex.from_pandas(obj, nan_as_null=nan_as_null)
    elif isinstance(obj, pd.RangeIndex):
        return cudf.core.index.RangeIndex(
            start=obj.start, stop=obj.stop, step=obj.step, name=obj.name
        )
    elif isinstance(obj, pd.Index):
        return cudf.Index.from_pandas(obj, nan_as_null=nan_as_null)
    elif isinstance(obj, pd.CategoricalDtype):
        return cudf.CategoricalDtype.from_pandas(obj)
    else:
        raise TypeError(
            "from_pandas only accepts Pandas Dataframes, Series, "
            "Index, RangeIndex and MultiIndex objects. "
            "Got %s" % type(obj)
        )


def merge(left, right, *args, **kwargs):
    return left.merge(right, *args, **kwargs)


# a bit of fanciness to inject docstring with left parameter
merge_doc = DataFrame.merge.__doc__
if merge_doc is not None:
    idx = merge_doc.find("right")
    merge.__doc__ = "".join(
        [merge_doc[:idx], "\n\tleft : DataFrame\n\t", merge_doc[idx:]]
    )


def _align_indices(lhs, rhs):
    """
    Internal util to align the indices of two DataFrames. Returns a tuple of
    the aligned dataframes, or the original arguments if the indices are the
    same, or if rhs isn't a DataFrame.
    """
    lhs_out, rhs_out = lhs, rhs
    if isinstance(rhs, DataFrame) and not lhs.index.equals(rhs.index):
        df = lhs.merge(
            rhs,
            sort=True,
            how="outer",
            left_index=True,
            right_index=True,
            suffixes=("_x", "_y"),
        )
        df = df.sort_index()
        lhs_out = DataFrame(index=df.index)
        rhs_out = DataFrame(index=df.index)
        common = set(lhs.columns) & set(rhs.columns)
        common_x = set(["{}_x".format(x) for x in common])
        common_y = set(["{}_y".format(x) for x in common])
        for col in df.columns:
            if col in common_x:
                lhs_out[col[:-2]] = df[col]
            elif col in common_y:
                rhs_out[col[:-2]] = df[col]
            elif col in lhs:
                lhs_out[col] = df[col]
            elif col in rhs:
                rhs_out[col] = df[col]

    return lhs_out, rhs_out


def _setitem_with_dataframe(
    input_df: DataFrame,
    replace_df: DataFrame,
    input_cols: Any = None,
    mask: Optional[cudf.core.column.ColumnBase] = None,
):
    """
    This function sets item dataframes relevant columns with replacement df
    :param input_df: Dataframe to be modified inplace
    :param replace_df: Replacement DataFrame to replace values with
    :param input_cols: columns to replace in the input dataframe
    :param mask: boolean mask in case of masked replacing
    """

    if input_cols is None:
        input_cols = input_df.columns

    if len(input_cols) != len(replace_df.columns):
        raise ValueError(
            "Number of Input Columns must be same replacement Dataframe"
        )

    if len(input_df) != 0 and not input_df.index.equals(replace_df.index):
        replace_df = replace_df.reindex(input_df.index)

    for col_1, col_2 in zip(input_cols, replace_df.columns):
        if col_1 in input_df.columns:
            if mask is not None:
                input_df._data[col_1][mask] = column.as_column(
                    replace_df[col_2]
                )
            else:
                input_df._data[col_1] = column.as_column(replace_df[col_2])
        else:
            if mask is not None:
                raise ValueError("Can not insert new column with a bool mask")
            else:
                # handle append case
                input_df.insert(len(input_df._data), col_1, replace_df[col_2])


def extract_col(df, col):
    """
    Extract column from dataframe `df` with their name `col`.
    If `col` is index and there are no columns with name `index`,
    then this will return index column.
    """
    try:
        return df._data[col]
    except KeyError:
        if (
            col == "index"
            and col not in df.index._data
            and not isinstance(df.index, MultiIndex)
        ):
            return df.index._data.columns[0]
        return df.index._data[col]


def _get_union_of_indices(indexes):
    if len(indexes) == 1:
        return indexes[0]
    else:
        merged_index = cudf.core.index.GenericIndex._concat(indexes)
        merged_index = merged_index.drop_duplicates()
        _, inds = merged_index._values.sort_by_values()
        return merged_index.take(inds)


def _get_union_of_series_names(series_list):
    names_list = []
    unnamed_count = 0
    for series in series_list:
        if series.name is None:
            names_list.append(f"Unnamed {unnamed_count}")
            unnamed_count += 1
        else:
            names_list.append(series.name)
    if unnamed_count == len(series_list):
        names_list = [*range(len(series_list))]

    return names_list


def _get_host_unique(array):
    if isinstance(
        array, (cudf.Series, cudf.Index, cudf.core.column.ColumnBase)
    ):
        return array.unique.to_pandas()
    elif isinstance(array, (str, numbers.Number)):
        return [array]
    else:
        return set(array)


def _drop_columns(df: DataFrame, columns: Iterable, errors: str):
    for c in columns:
        try:
            df._drop_column(c)
        except KeyError as e:
            if errors == "ignore":
                pass
            else:
                raise e


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
        if all(is_numeric_dtype(col.dtype) for col in cols):
            dtypes[idx] = find_common_type([col.dtype for col in cols])
        # If all categorical dtypes, combine the categories
        elif all(
            isinstance(col, cudf.core.column.CategoricalColumn) for col in cols
        ):
            # Combine and de-dupe the categories
            categories[idx] = (
                cudf.Series(concat_columns([col.categories for col in cols]))
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
                        ._set_categories(categories[idx], is_unique=True,)
                        .codes
                    )
                cols[idx] = cols[idx].astype(dtype)


def _reassign_categories(categories, cols, col_idxs):
    for name, idx in zip(cols, col_idxs):
        if idx in categories:
            cols[name] = build_categorical_column(
                categories=categories[idx],
                codes=build_column(
                    cols[name].base_data, dtype=cols[name].dtype
                ),
                mask=cols[name].base_mask,
                offset=cols[name].offset,
                size=cols[name].size,
            )
