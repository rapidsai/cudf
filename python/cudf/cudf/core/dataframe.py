# Copyright (c) 2018-2024, NVIDIA CORPORATION.

from __future__ import annotations

import functools
import inspect
import itertools
import numbers
import os
import pickle
import re
import sys
import textwrap
import warnings
from collections import abc, defaultdict
from collections.abc import Iterator
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import cupy
import numba
import numpy as np
import pandas as pd
import pyarrow as pa
from nvtx import annotate
from pandas.io.formats import console
from pandas.io.formats.printing import pprint_thing
from typing_extensions import Self, assert_never

import cudf
import cudf.core.common
from cudf import _lib as libcudf
from cudf._typing import ColumnLike, Dtype, NotImplementedType
from cudf.api.extensions import no_default
from cudf.api.types import (
    _is_scalar_or_zero_d_array,
    is_bool_dtype,
    is_datetime_dtype,
    is_dict_like,
    is_dtype_equal,
    is_list_like,
    is_numeric_dtype,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
)
from cudf.core import column, df_protocol, indexing_utils, reshape
from cudf.core._compat import PANDAS_LT_300
from cudf.core.abc import Serializable
from cudf.core.column import (
    CategoricalColumn,
    ColumnBase,
    StructColumn,
    as_column,
    build_categorical_column,
    build_column,
    column_empty,
    concat_columns,
)
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.copy_types import BooleanMask
from cudf.core.groupby.groupby import DataFrameGroupBy, groupby_doc_template
from cudf.core.index import BaseIndex, RangeIndex, _index_from_data, as_index
from cudf.core.indexed_frame import (
    IndexedFrame,
    _FrameIndexer,
    _get_label_range_or_mask,
    _indices_from_labels,
    doc_reset_index_template,
)
from cudf.core.join import Merge, MergeSemi
from cudf.core.missing import NA
from cudf.core.multiindex import MultiIndex
from cudf.core.resample import DataFrameResampler
from cudf.core.series import Series
from cudf.core.udf.row_function import _get_row_kernel
from cudf.errors import MixedTypeError
from cudf.utils import applyutils, docutils, ioutils, queryutils
from cudf.utils.docutils import copy_docstring
from cudf.utils.dtypes import (
    can_convert_to_column,
    cudf_dtype_from_pydata_dtype,
    find_common_type,
    is_column_like,
    min_scalar_type,
    numeric_normalize_types,
)
from cudf.utils.nvtx_annotation import _cudf_nvtx_annotate
from cudf.utils.utils import GetAttrGetItemMixin, _external_only_api

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

_numeric_reduction_ops = (
    "mean",
    "min",
    "max",
    "sum",
    "product",
    "prod",
    "std",
    "var",
    "kurtosis",
    "kurt",
    "skew",
)


def _shape_mismatch_error(x, y):
    raise ValueError(
        f"shape mismatch: value array of shape {x} "
        f"could not be broadcast to indexing result of "
        f"shape {y}"
    )


class _DataFrameIndexer(_FrameIndexer):
    def __getitem__(self, arg):
        if (
            isinstance(self._frame.index, MultiIndex)
            or self._frame._data.multiindex
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

    @_cudf_nvtx_annotate
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
            all_numeric = all(is_numeric_dtype(t) for t in dtypes)
            if all_numeric or (
                len(dtypes) and all(t == dtypes[0] for t in dtypes)
            ):
                return True
            if isinstance(arg[1], tuple):
                return True
        if ncols == 1:
            if type(arg[1]) is slice:
                return False
            if isinstance(arg[1], tuple):
                return len(arg[1]) == df._data.nlevels
            if not (is_list_like(arg[1]) or is_column_like(arg[1])):
                return True
        return False

    @_cudf_nvtx_annotate
    def _downcast_to_series(self, df, arg):
        """
        "Downcast" from a DataFrame to a Series
        based on Pandas indexing rules
        """
        nrows, ncols = df.shape
        # determine the axis along which the Series is taken:
        if nrows == 1 and ncols == 1:
            if is_scalar(arg[0]) and (
                is_scalar(arg[1])
                or (df._data.multiindex and arg[1] in df._column_names)
            ):
                return df[df._column_names[0]].iloc[0]
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

    @_cudf_nvtx_annotate
    def _getitem_scalar(self, arg):
        return self._frame[arg[1]].loc[arg[0]]

    @_cudf_nvtx_annotate
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
                    row_arg = arg[0]
                elif is_scalar(arg):
                    row_arg = (arg,)
                else:
                    row_arg = arg
                result = columns_df.index._get_row_major(columns_df, row_arg)
                if (
                    len(result) == 1
                    and isinstance(arg, tuple)
                    and len(arg) > 1
                    and is_scalar(arg[1])
                ):
                    return result._data.columns[0].element_indexing(0)
                return result
        else:
            if isinstance(arg[0], slice):
                out = _get_label_range_or_mask(
                    columns_df.index, arg[0].start, arg[0].stop, arg[0].step
                )
                if isinstance(out, slice):
                    df = columns_df._slice(out)
                else:
                    df = columns_df._apply_boolean_mask(
                        BooleanMask.from_column_unchecked(
                            cudf.core.column.as_column(out)
                        )
                    )
            else:
                tmp_arg = arg
                if is_scalar(arg[0]):
                    # If a scalar, there is possibility of having duplicates.
                    # Join would get all the duplicates. So, converting it to
                    # an array kind.
                    if cudf.get_option("mode.pandas_compatible"):
                        if any(
                            c.dtype != columns_df._columns[0].dtype
                            for c in columns_df._columns
                        ):
                            raise TypeError(
                                "All columns need to be of same type, please "
                                "typecast to common dtype."
                            )
                    tmp_arg = ([tmp_arg[0]], tmp_arg[1])
                if len(tmp_arg[0]) == 0:
                    return columns_df._empty_like(keep_index=True)
                tmp_arg = (
                    as_column(
                        tmp_arg[0],
                        dtype=self._frame.index.dtype
                        if isinstance(
                            self._frame.index.dtype, cudf.CategoricalDtype
                        )
                        else None,
                    ),
                    tmp_arg[1],
                )

                if is_bool_dtype(tmp_arg[0].dtype):
                    df = columns_df._apply_boolean_mask(
                        BooleanMask(tmp_arg[0], len(columns_df))
                    )
                else:
                    tmp_col_name = str(uuid4())
                    cantor_name = "_" + "_".join(
                        map(str, columns_df._data.names)
                    )
                    if columns_df._data.multiindex:
                        # column names must be appropriate length tuples
                        extra = tuple(
                            "" for _ in range(columns_df._data.nlevels - 1)
                        )
                        tmp_col_name = (tmp_col_name, *extra)
                        cantor_name = (cantor_name, *extra)
                    other_df = DataFrame(
                        {
                            tmp_col_name: column.as_column(
                                range(len(tmp_arg[0]))
                            )
                        },
                        index=as_index(tmp_arg[0]),
                    )
                    columns_df[cantor_name] = column.as_column(
                        range(len(columns_df))
                    )
                    df = other_df.join(columns_df, how="inner")
                    # as join is not assigning any names to index,
                    # update it over here
                    df.index.name = columns_df.index.name
                    if not isinstance(
                        df.index, MultiIndex
                    ) and is_numeric_dtype(df.index.dtype):
                        # Preserve the original index type.
                        df.index = df.index.astype(self._frame.index.dtype)
                    df = df.sort_values(by=[tmp_col_name, cantor_name])
                    df.drop(columns=[tmp_col_name, cantor_name], inplace=True)
                    # There were no indices found
                    if len(df) == 0:
                        raise KeyError(arg)

        # Step 3: Downcast
        if self._can_downcast_to_series(df, arg):
            return self._downcast_to_series(df, arg)
        return df

    @_cudf_nvtx_annotate
    def _setitem_tuple_arg(self, key, value):
        if (
            isinstance(self._frame.index, MultiIndex)
            or self._frame._data.multiindex
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
                if is_scalar(key[0]):
                    arr = [key[0]]
                else:
                    arr = key[0]
                idx = cudf.Index(arr)
            if is_scalar(value):
                length = len(idx) if idx is not None else 1
                value = as_column(value, length=length)

            new_col = cudf.Series(value, index=idx)
            if len(self._frame.index) != 0:
                new_col = new_col._align_to_index(
                    self._frame.index, how="right"
                )

            if len(self._frame.index) == 0:
                self._frame.index = (
                    idx if idx is not None else cudf.RangeIndex(len(new_col))
                )
            self._frame._data.insert(key[1], new_col)
        else:
            if is_scalar(value):
                for col in columns_df._column_names:
                    self._frame[col].loc[key[0]] = value

            elif isinstance(value, cudf.DataFrame):
                if value.shape != self._frame.loc[key[0]].shape:
                    _shape_mismatch_error(
                        value.shape,
                        self._frame.loc[key[0]].shape,
                    )
                value_column_names = set(value._column_names)
                scatter_map = _indices_from_labels(self._frame, key[0])
                for col in columns_df._column_names:
                    columns_df[col][scatter_map] = (
                        value._data[col] if col in value_column_names else NA
                    )

            else:
                value = cupy.asarray(value)
                if cupy.ndim(value) == 2:
                    # If the inner dimension is 1, it's broadcastable to
                    # all columns of the dataframe.
                    indexed_shape = columns_df.loc[key[0]].shape
                    if value.shape[1] == 1:
                        if value.shape[0] != indexed_shape[0]:
                            _shape_mismatch_error(value.shape, indexed_shape)
                        for i, col in enumerate(columns_df._column_names):
                            self._frame[col].loc[key[0]] = value[:, 0]
                    else:
                        if value.shape != indexed_shape:
                            _shape_mismatch_error(value.shape, indexed_shape)
                        for i, col in enumerate(columns_df._column_names):
                            self._frame[col].loc[key[0]] = value[:, i]
                else:
                    # handle cases where value is 1d object:
                    # If the key on column axis is a scalar, we indexed
                    # a single column; The 1d value should assign along
                    # the columns.
                    if is_scalar(key[1]):
                        for col in columns_df._column_names:
                            self._frame[col].loc[key[0]] = value
                    # Otherwise, there are two situations. The key on row axis
                    # can be a scalar or 1d. In either of the situation, the
                    # ith element in value corresponds to the ith row in
                    # the indexed object.
                    # If the key is 1d, a broadcast will happen.
                    else:
                        for i, col in enumerate(columns_df._column_names):
                            self._frame[col].loc[key[0]] = value[i]


class _DataFrameIlocIndexer(_DataFrameIndexer):
    """
    For selection by index.
    """

    _frame: DataFrame

    def __getitem__(self, arg):
        (
            row_key,
            (
                col_is_scalar,
                column_names,
            ),
        ) = indexing_utils.destructure_dataframe_iloc_indexer(arg, self._frame)
        row_spec = indexing_utils.parse_row_iloc_indexer(
            row_key, len(self._frame)
        )
        ca = self._frame._data
        index = self._frame.index
        if col_is_scalar:
            s = Series._from_data(
                data=ColumnAccessor(
                    {key: ca._data[key] for key in column_names},
                    multiindex=ca.multiindex,
                    level_names=ca.level_names,
                ),
                index=index,
            )
            return s._getitem_preprocessed(row_spec)
        if column_names != list(self._frame._column_names):
            frame = self._frame._from_data(
                data=ColumnAccessor(
                    {key: ca._data[key] for key in column_names},
                    multiindex=ca.multiindex,
                    level_names=ca.level_names,
                ),
                index=index,
            )
        else:
            frame = self._frame
        if isinstance(row_spec, indexing_utils.MapIndexer):
            return frame._gather(row_spec.key, keep_index=True)
        elif isinstance(row_spec, indexing_utils.MaskIndexer):
            return frame._apply_boolean_mask(row_spec.key, keep_index=True)
        elif isinstance(row_spec, indexing_utils.SliceIndexer):
            return frame._slice(row_spec.key)
        elif isinstance(row_spec, indexing_utils.ScalarIndexer):
            result = frame._gather(row_spec.key, keep_index=True)
            # Attempt to turn into series.
            try:
                # Behaviour difference from pandas, which will merrily
                # turn any heterogeneous set of columns into a series if
                # you only ask for one row.
                new_name = result.index[0]
                result = Series._concat(
                    [result[name] for name in column_names],
                    index=result.keys(),
                )
                result.name = new_name
                return result
            except TypeError:
                # Couldn't find a common type, Hence:
                # Raise in pandas compatibility mode,
                # or just return a 1xN dataframe otherwise
                if cudf.get_option("mode.pandas_compatible"):
                    raise TypeError(
                        "All columns need to be of same type, please "
                        "typecast to common dtype."
                    )
                return result
        elif isinstance(row_spec, indexing_utils.EmptyIndexer):
            return frame._empty_like(keep_index=True)
        assert_never(row_spec)

    @_cudf_nvtx_annotate
    def _setitem_tuple_arg(self, key, value):
        columns_df = self._frame._from_data(
            self._frame._data.select_by_index(key[1]), self._frame._index
        )

        if is_scalar(value):
            for col in columns_df._column_names:
                self._frame[col].iloc[key[0]] = value

        elif isinstance(value, cudf.DataFrame):
            if value.shape != self._frame.iloc[key[0]].shape:
                _shape_mismatch_error(
                    value.shape,
                    self._frame.loc[key[0]].shape,
                )
            value_column_names = set(value._column_names)
            for col in columns_df._column_names:
                columns_df[col][key[0]] = (
                    value._data[col] if col in value_column_names else NA
                )

        else:
            # TODO: consolidate code path with identical counterpart
            # in `_DataFrameLocIndexer._setitem_tuple_arg`
            value = cupy.asarray(value)
            if cupy.ndim(value) == 2:
                indexed_shape = columns_df.iloc[key[0]].shape
                if value.shape[1] == 1:
                    if value.shape[0] != indexed_shape[0]:
                        _shape_mismatch_error(value.shape, indexed_shape)
                    for i, col in enumerate(columns_df._column_names):
                        self._frame[col].iloc[key[0]] = value[:, 0]
                else:
                    if value.shape != indexed_shape:
                        _shape_mismatch_error(value.shape, indexed_shape)
                    for i, col in enumerate(columns_df._column_names):
                        self._frame._data[col][key[0]] = value[:, i]
            else:
                if is_scalar(key[1]):
                    for col in columns_df._column_names:
                        self._frame[col].iloc[key[0]] = value
                else:
                    for i, col in enumerate(columns_df._column_names):
                        self._frame[col].iloc[key[0]] = value[i]


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
        id            datetimes
    0    0  2018-10-07 12:00:00
    1    1  2018-10-07 12:00:01
    2    2  2018-10-07 12:00:02
    3    3  2018-10-07 12:00:03
    4    4  2018-10-07 12:00:04

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
    _groupby = DataFrameGroupBy
    _resampler = DataFrameResampler

    @_cudf_nvtx_annotate
    def __init__(
        self,
        data=None,
        index=None,
        columns=None,
        dtype=None,
        nan_as_null=no_default,
    ):
        super().__init__()
        if nan_as_null is no_default:
            nan_as_null = not cudf.get_option("mode.pandas_compatible")

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
                    column_names=columns, index=index, deep=False, inplace=True
                )
                if isinstance(
                    columns, (range, pd.RangeIndex, cudf.RangeIndex)
                ):
                    self._data.rangeindex = True
            else:
                self._data = data._data
                self._data.rangeindex = True
        elif isinstance(data, (cudf.Series, pd.Series)):
            if isinstance(data, pd.Series):
                data = cudf.Series.from_pandas(data, nan_as_null=nan_as_null)

            # Series.name is not None and Series.name in columns
            #   -> align
            # Series.name is not None and Series.name not in columns
            #   -> return empty DataFrame
            # Series.name is None and no columns
            #   -> return 1 column DataFrame
            # Series.name is None and columns
            #   -> return 1 column DataFrame if len(columns) in {0, 1}
            if data.name is None and columns is not None:
                if len(columns) > 1:
                    raise ValueError(
                        "Length of columns must be less than 2 if "
                        f"{type(data).__name__}.name is None."
                    )
                name = columns[0]
            else:
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
                rangeindex = isinstance(
                    columns, (range, pd.RangeIndex, cudf.RangeIndex)
                )
                label_dtype = getattr(columns, "dtype", None)
                self._data = ColumnAccessor(
                    {
                        k: column.column_empty(
                            len(self), dtype="object", masked=True
                        )
                        for k in columns
                    },
                    level_names=tuple(columns.names)
                    if isinstance(columns, pd.Index)
                    else None,
                    rangeindex=rangeindex,
                    label_dtype=label_dtype,
                )
        elif isinstance(data, ColumnAccessor):
            raise TypeError(
                "Use cudf.Series._from_data for constructing a Series from "
                "ColumnAccessor"
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
            self._index = new_df._index
            self._check_data_index_length_match()
        elif hasattr(data, "__array_interface__"):
            arr_interface = data.__array_interface__
            if len(arr_interface["descr"]) == 1:
                # not record arrays
                new_df = self._from_arrays(data, index=index, columns=columns)
            else:
                new_df = self.from_records(data, index=index, columns=columns)
            self._data = new_df._data
            self._index = new_df._index
            self._check_data_index_length_match()
        else:
            if isinstance(data, Iterator):
                data = list(data)
            if is_list_like(data):
                if len(data) > 0 and is_scalar(data[0]):
                    if columns is not None:
                        label_dtype = getattr(columns, "dtype", None)
                        data = dict(zip(columns, [data]))
                        rangeindex = isinstance(
                            columns, (range, pd.RangeIndex, cudf.RangeIndex)
                        )
                    else:
                        data = dict(enumerate([data]))
                        rangeindex = True
                        label_dtype = None
                    new_df = DataFrame(data=data, index=index)

                    self._data = new_df._data
                    self._index = new_df._index
                    self._data._level_names = (
                        tuple(columns.names)
                        if isinstance(columns, pd.Index)
                        else self._data._level_names
                    )
                    self._data.rangeindex = rangeindex
                    self._data.label_dtype = (
                        cudf.dtype(label_dtype)
                        if label_dtype is not None
                        else None
                    )
                elif len(data) > 0 and isinstance(data[0], Series):
                    self._init_from_series_list(
                        data=data, columns=columns, index=index
                    )
                else:
                    self._init_from_list_like(
                        data, index=index, columns=columns
                    )
                self._check_data_index_length_match()
            else:
                if not is_dict_like(data):
                    raise TypeError("data must be list or dict-like")

                self._init_from_dict_like(
                    data, index=index, columns=columns, nan_as_null=nan_as_null
                )
                self._check_data_index_length_match()

        if dtype:
            self._data = self.astype(dtype)._data

        self._data.multiindex = self._data.multiindex or isinstance(
            columns, pd.MultiIndex
        )

    @_cudf_nvtx_annotate
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
                        f"Length of values ({data_length}) does "
                        f"not match length of index ({index_length})"
                    )

            final_index = as_index(index)

        series_lengths = list(map(len, data))
        data = numeric_normalize_types(*data)
        if series_lengths.count(series_lengths[0]) == len(series_lengths):
            # Calculating the final dataframe columns by
            # getting union of all `index` of the Series objects.
            final_columns = _get_union_of_indices([d.index for d in data])
            if isinstance(final_columns, cudf.RangeIndex):
                self._data.rangeindex = True

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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                concat_df = cudf.concat(data, axis=1)

            cols = concat_df._data.to_pandas_index()
            if cols.dtype == "object":
                concat_df.columns = cols.astype("str")

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
            self._data._level_names = (
                tuple(columns.names)
                if isinstance(columns, pd.Index)
                else self._data._level_names
            )
            self._data = self._data.select_by_label(columns)
            self._data.rangeindex = isinstance(
                columns, (range, cudf.RangeIndex, pd.RangeIndex)
            )
        else:
            self._data.rangeindex = True

    @_cudf_nvtx_annotate
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
        elif len(data) > 0 and isinstance(data[0], pd.Interval):
            data = DataFrame.from_pandas(pd.DataFrame(data))
            self._data = data._data
        elif any(
            not isinstance(col, (abc.Iterable, abc.Sequence)) for col in data
        ):
            raise TypeError("Inputs should be an iterable or sequence.")
        elif len(data) > 0 and not can_convert_to_column(data[0]):
            raise ValueError("Must pass 2-d input.")
        else:
            if (
                len(data) > 0
                and columns is None
                and isinstance(data[0], tuple)
                and hasattr(data[0], "_fields")
            ):
                # pandas behavior is to use the fields from the first
                # namedtuple as the column names
                columns = data[0]._fields

            data = list(itertools.zip_longest(*data))

            if columns is not None and len(data) == 0:
                data = [
                    cudf.core.column.column_empty(row_count=0, dtype=None)
                    for _ in columns
                ]

            for col_name, col in enumerate(data):
                self._data[col_name] = column.as_column(col)
            self._data.rangeindex = True

        if columns is not None:
            if len(columns) != len(data):
                raise ValueError(
                    f"Shape of passed values is ({len(index)}, {len(data)}), "
                    f"indices imply ({len(index)}, {len(columns)})."
                )

            self.columns = columns
            self._data.rangeindex = isinstance(
                columns, (range, pd.RangeIndex, cudf.RangeIndex)
            )
            self._data.label_dtype = getattr(columns, "dtype", None)

    @_cudf_nvtx_annotate
    def _init_from_dict_like(
        self, data, index=None, columns=None, nan_as_null=None
    ):
        label_dtype = None
        if columns is not None:
            label_dtype = getattr(columns, "dtype", None)
            # remove all entries in data that are not in columns,
            # inserting new empty columns for entries in columns that
            # are not in data
            if any(c in data for c in columns):
                # Let the downstream logic determine the length of the
                # empty columns here
                empty_column = lambda: None  # noqa: E731
            else:
                # If keys is empty, none of the data keys match the
                # columns, so we need to create an empty DataFrame. To
                # match pandas, the size of the dataframe must match
                # the provided index, so we need to return a masked
                # array of nulls if an index is given.
                empty_column = functools.partial(
                    cudf.core.column.column_empty,
                    row_count=(0 if index is None else len(index)),
                    masked=index is not None,
                )

            data = {
                c: data[c] if c in data else empty_column() for c in columns
            }

        data, index = self._align_input_series_indices(data, index=index)

        if index is None:
            num_rows = 0
            if data:
                keys, values, lengths = zip(
                    *(
                        (k, v, 1)
                        if is_scalar(v)
                        else (
                            k,
                            vc := as_column(v, nan_as_null=nan_as_null),
                            len(vc),
                        )
                        for k, v in data.items()
                    )
                )
                data = dict(zip(keys, values))
                try:
                    (num_rows,) = (set(lengths) - {1}) or {1}
                except ValueError:
                    raise ValueError("All arrays must be the same length")

            self._index = RangeIndex(0, num_rows)
        else:
            self._index = as_index(index)

        if len(data):
            self._data.multiindex = True
            for i, col_name in enumerate(data):
                self._data.multiindex = self._data.multiindex and isinstance(
                    col_name, tuple
                )
                self._insert(
                    i,
                    col_name,
                    data[col_name],
                    nan_as_null=nan_as_null,
                )
        self._data._level_names = (
            tuple(columns.names)
            if isinstance(columns, pd.Index)
            else self._data._level_names
        )
        self._data.label_dtype = label_dtype

    @classmethod
    def _from_data(
        cls,
        data: MutableMapping,
        index: Optional[BaseIndex] = None,
        columns: Any = None,
    ) -> DataFrame:
        out = super()._from_data(data=data, index=index)
        if columns is not None:
            out.columns = columns
        return out

    @staticmethod
    @_cudf_nvtx_annotate
    def _align_input_series_indices(data, index):
        data = data.copy()

        input_series = [
            Series(val)
            for val in data.values()
            if isinstance(val, (pd.Series, Series, dict))
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
                if isinstance(val, (pd.Series, Series, dict)):
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
    @_cudf_nvtx_annotate
    def shape(self):
        """Returns a tuple representing the dimensionality of the DataFrame."""
        return self._num_rows, self._num_columns

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
        datetime    datetime64[ns]
        string              object
        dtype: object
        """
        return pd.Series(self._dtypes, dtype="object")

    @property
    def ndim(self) -> int:
        """Dimension of the data. DataFrame ndim is always 2."""
        return 2

    def __dir__(self):
        # Add the columns of the DataFrame to the dir output.
        o = set(dir(type(self)))
        o.update(self.__dict__)
        o.update(
            c
            for c in self._column_names
            if isinstance(c, str) and c.isidentifier()
        )
        return list(o)

    def __setattr__(self, key, col):
        try:
            # Preexisting attributes may be set. We cannot rely on checking the
            # `_PROTECTED_KEYS` because we must also allow for settable
            # properties, and we must call object.__getattribute__ to bypass
            # the `__getitem__` behavior inherited from `GetAttrGetItemMixin`.
            object.__getattribute__(self, key)
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

        except RuntimeError as e:
            # TODO: This allows setting properties that are marked as forbidden
            # for internal usage. It is necessary because the __getattribute__
            # call in the try block will trigger the error. We should see if
            # setting these variables can also always be disabled
            if "External-only API" not in str(e):
                raise
            super().__setattr__(key, col)
        else:
            super().__setattr__(key, col)

    @_cudf_nvtx_annotate
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
        >>> df = cudf.DataFrame({
        ...     'a': list(range(10)),
        ...     'b': list(range(10)),
        ...     'c': list(range(10)),
        ... })

        Get first 4 rows of all columns.

        >>> df[:4]
           a  b  c
        0  0  0  0
        1  1  1  1
        2  2  2  2
        3  3  3  3

        Get last 5 rows of all columns.

        >>> df[-5:]
           a  b  c
        5  5  5  5
        6  6  6  6
        7  7  7  7
        8  8  8  8
        9  9  9  9

        Get columns a and c.

        >>> df[['a', 'c']]
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

        Return the rows specified in the boolean mask.

        >>> df[[True, False, True, False, True,
        ...     False, True, False, True, False]]
           a  b  c
        0  0  0  0
        2  2  2  2
        4  4  4  4
        6  6  6  6
        8  8  8  8
        """
        if _is_scalar_or_zero_d_array(arg) or isinstance(arg, tuple):
            return self._get_columns_by_label(arg, downcast=True)

        elif isinstance(arg, slice):
            return self._slice(arg)

        elif can_convert_to_column(arg):
            mask = arg
            if is_list_like(mask):
                dtype = None
                mask = pd.Series(mask, dtype=dtype)
            if mask.dtype == "bool":
                return self._apply_boolean_mask(BooleanMask(mask, len(self)))
            else:
                return self._get_columns_by_label(mask)
        elif isinstance(arg, DataFrame):
            return self.where(arg)
        else:
            raise TypeError(
                f"__getitem__ on type {type(arg)} is not supported"
            )

    @_cudf_nvtx_annotate
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
                    scatter_map = arg._data[col_name]
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
                    if not is_scalar(value) and len(self) == 0:
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
                        self._data[arg] = as_column(value, length=len(self))
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
                if isinstance(value, (cupy.ndarray, np.ndarray)):
                    _setitem_with_dataframe(
                        input_df=self,
                        replace_df=cudf.DataFrame(value),
                        input_cols=arg,
                        mask=None,
                        ignore_index=True,
                    )
                elif isinstance(value, DataFrame):
                    _setitem_with_dataframe(
                        input_df=self,
                        replace_df=value,
                        input_cols=arg,
                        mask=None,
                    )
                else:
                    for col in arg:
                        if is_scalar(value):
                            self._data[col] = as_column(
                                value, length=len(self)
                            )
                        else:
                            self._data[col] = column.as_column(value)

        else:
            raise TypeError(
                f"__setitem__ on type {type(arg)} is not supported"
            )

    def __delitem__(self, name):
        self._drop_column(name)

    @_cudf_nvtx_annotate
    def memory_usage(self, index=True, deep=False):
        mem_usage = [col.memory_usage for col in self._data.columns]
        names = [str(name) for name in self._data.names]
        if index:
            mem_usage.append(self._index.memory_usage())
            names.append("Index")
        return Series._from_data(
            data={None: as_column(mem_usage)},
            index=as_index(names),
        )

    @_cudf_nvtx_annotate
    def __array_function__(self, func, types, args, kwargs):
        if "out" in kwargs or not all(
            issubclass(t, (Series, DataFrame)) for t in types
        ):
            return NotImplemented

        try:
            if func.__name__ in {"any", "all"}:
                # NumPy default for `axis` is
                # different from `cudf`/`pandas`
                # hence need this special handling.
                kwargs.setdefault("axis", None)
            if cudf_func := getattr(self.__class__, func.__name__, None):
                out = cudf_func(*args, **kwargs)
                # The dot product of two DataFrames returns an array in pandas.
                if (
                    func is np.dot
                    and isinstance(args[0], (DataFrame, pd.DataFrame))
                    and isinstance(args[1], (DataFrame, pd.DataFrame))
                ):
                    return out.values
                return out
        except Exception:
            # The rare instance where a "silent" failure is preferable. Except
            # in the (highly unlikely) case that some other library
            # interoperates with cudf objects, the result will be that numpy
            # raises a TypeError indicating that the operation is not
            # implemented, which is much friendlier than an arbitrary internal
            # cudf error.
            pass
        return NotImplemented

    # The _get_numeric_data method is necessary for dask compatibility.
    @_cudf_nvtx_annotate
    def _get_numeric_data(self):
        """Return a dataframe with only numeric data types"""
        columns = [
            c
            for c, dt in self.dtypes.items()
            if dt != object and not isinstance(dt, cudf.CategoricalDtype)
        ]
        return self[columns]

    @_cudf_nvtx_annotate
    def assign(self, **kwargs: Union[Callable[[Self], Any], Any]):
        """
        Assign columns to DataFrame from keyword arguments.

        Parameters
        ----------
        **kwargs: dict mapping string column names to values
            The value for each key can either be a literal column (or
            something that can be converted to a column), or
            a callable of one argument that will be given the
            dataframe as an argument and should return the new column
            (without modifying the input argument).
            Columns are added in-order, so callables can refer to
            column names constructed in the assignment.

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
        new_df = self.copy(deep=False)
        for k, v in kwargs.items():
            new_df[k] = v(new_df) if callable(v) else v
        return new_df

    @classmethod
    @_cudf_nvtx_annotate
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
                # For pandas compatibility, we also try to handle the case
                # where some column names are strings and others are ints. Just
                # assume that everything that isn't a str is numerical, we
                # can't sort anything else.
                try:
                    str_names = sorted(n for n in names if isinstance(n, str))
                    non_str_names = sorted(
                        n for n in names if not isinstance(n, str)
                    )
                    names = non_str_names + str_names
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
        # The position of the first table column in each
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
                table_index = DataFrame._from_data(
                    data=dict(
                        zip(
                            indices[:first_data_column_position],
                            cols[:first_data_column_position],
                        )
                    )
                )
            tables.append(
                DataFrame._from_data(
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
            out._index = cudf.core.index.Index._concat(
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
            if not isinstance(out._index, MultiIndex) and isinstance(
                out._index.dtype, cudf.CategoricalDtype
            ):
                out = out.set_index(
                    cudf.core.index.as_index(out.index._values)
                )
        for name, col in out._data.items():
            out._data[name] = col._with_type_metadata(
                tables[0]._data[name].dtype
            )

        # Reassign index and column names
        if objs[0]._data.multiindex:
            out._set_columns_like(objs[0]._data)
        else:
            out.columns = names
        if not ignore_index:
            out._index.name = objs[0]._index.name
            out._index.names = objs[0]._index.names

        return out

    def astype(
        self,
        dtype,
        copy: bool = False,
        errors: Literal["raise", "ignore"] = "raise",
    ):
        if is_dict_like(dtype):
            if len(set(dtype.keys()) - set(self._data.names)) > 0:
                raise KeyError(
                    "Only a column name can be used for the "
                    "key in a dtype mappings argument."
                )
        else:
            dtype = {cc: dtype for cc in self._data.names}
        return super().astype(dtype, copy, errors)

    def _clean_renderable_dataframe(self, output):
        """
        This method takes in partial/preprocessed dataframe
        and returns correct representation of it with correct
        dimensions (rows x columns)
        """

        max_rows = pd.get_option("display.max_rows")
        min_rows = pd.get_option("display.min_rows")
        max_cols = pd.get_option("display.max_columns")
        max_colwidth = pd.get_option("display.max_colwidth")
        show_dimensions = pd.get_option("display.show_dimensions")
        if pd.get_option("display.expand_frame_repr"):
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
            if isinstance(
                df._data[col].dtype, (cudf.StructDtype, cudf.ListDtype)
            ):
                # TODO we need to handle this
                pass
            elif df._data[col].has_nulls():
                fill_value = (
                    str(cudf.NaT)
                    if isinstance(
                        df._data[col],
                        (
                            cudf.core.column.DatetimeColumn,
                            cudf.core.column.TimeDeltaColumn,
                        ),
                    )
                    else str(cudf.NA)
                )

                df[col] = df._data[col].astype("str").fillna(fill_value)
            else:
                df[col] = df._data[col]

        return df

    def _get_renderable_dataframe(self):
        """
        Takes rows and columns from pandas settings or estimation from size.
        pulls quadrants based off of some known parameters then style for
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
            # In case of Empty DataFrame with index, Pandas prints
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

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                upper = cudf.concat([upper_left, upper_right], axis=1)
                lower = cudf.concat([lower_left, lower_right], axis=1)
                output = cudf.concat([upper, lower])

        output = self._clean_nulls_from_dataframe(output)
        output._index = output._index._clean_nulls_from_index()

        return output

    @_cudf_nvtx_annotate
    def __repr__(self):
        output = self._get_renderable_dataframe()
        return self._clean_renderable_dataframe(output)

    @_cudf_nvtx_annotate
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

    @_cudf_nvtx_annotate
    def _repr_latex_(self):
        return self._get_renderable_dataframe().to_pandas()._repr_latex_()

    @_cudf_nvtx_annotate
    def _get_columns_by_label(
        self, labels, *, downcast=False
    ) -> Self | Series:
        """
        Return columns of dataframe by `labels`

        If downcast is True, try and downcast from a DataFrame to a Series
        """
        ca = self._data.select_by_label(labels)
        if downcast:
            if is_scalar(labels):
                nlevels = 1
            elif isinstance(labels, tuple):
                nlevels = len(labels)
            if self._data.multiindex is False or nlevels == self._data.nlevels:
                out = self._constructor_sliced._from_data(
                    ca, index=self.index, name=labels
                )
                return out
        out = self.__class__._from_data(
            ca, index=self.index, columns=ca.to_pandas_index()
        )
        return out

    def _make_operands_and_index_for_binop(
        self,
        other: Any,
        fn: str,
        fill_value: Any = None,
        reflect: bool = False,
        can_reindex: bool = False,
    ) -> Tuple[
        Union[
            Dict[Optional[str], Tuple[ColumnBase, Any, bool, Any]],
            NotImplementedType,
        ],
        Optional[BaseIndex],
        bool,
    ]:
        lhs, rhs = self._data, other
        index = self._index
        fill_requires_key = False
        left_default: Any = False
        equal_columns = False
        can_use_self_column_name = True

        if _is_scalar_or_zero_d_array(other):
            rhs = {name: other for name in self._data}
            equal_columns = True
        elif isinstance(other, Series):
            if (
                not can_reindex
                and fn in cudf.utils.utils._EQUALITY_OPS
                and (
                    not self._data.to_pandas_index().equals(
                        other.index.to_pandas()
                    )
                )
            ):
                raise ValueError(
                    "Can only compare DataFrame & Series objects "
                    "whose columns & index are same respectively, "
                    "please reindex."
                )
            rhs = dict(zip(other.index.to_pandas(), other.values_host))
            # For keys in right but not left, perform binops between NaN (not
            # NULL!) and the right value (result is NaN).
            left_default = as_column(np.nan, length=len(self))
            equal_columns = other.index.to_pandas().equals(
                self._data.to_pandas_index()
            )
            can_use_self_column_name = (
                equal_columns
                or list(other._index._data.names) == self._data._level_names
            )
        elif isinstance(other, DataFrame):
            if (
                not can_reindex
                and fn in cudf.utils.utils._EQUALITY_OPS
                and (
                    not self.index.equals(other.index)
                    or not self._data.to_pandas_index().equals(
                        other._data.to_pandas_index()
                    )
                )
            ):
                raise ValueError(
                    "Can only compare identically-labeled DataFrame objects"
                )
            new_lhs, new_rhs = _align_indices(self, other)
            index = new_lhs._index
            lhs, rhs = new_lhs._data, new_rhs._data
            fill_requires_key = True
            # For DataFrame-DataFrame ops, always default to operating against
            # the fill value.
            left_default = fill_value
            equal_columns = self._column_names == other._column_names
            can_use_self_column_name = (
                equal_columns
                or self._data._level_names == other._data._level_names
            )
        elif isinstance(other, (dict, abc.Mapping)):
            # Need to fail early on host mapping types because we ultimately
            # convert everything to a dict.
            return NotImplemented, None, True

        if not isinstance(rhs, (dict, abc.Mapping)):
            return NotImplemented, None, True

        operands = {
            k: (
                v,
                rhs.get(k, fill_value),
                reflect,
                fill_value if (not fill_requires_key or k in rhs) else None,
            )
            for k, v in lhs.items()
        }

        if left_default is not False:
            for k, v in rhs.items():
                if k not in lhs:
                    operands[k] = (left_default, v, reflect, None)

        if not equal_columns:
            if isinstance(other, DataFrame):
                column_names_list = self._data.to_pandas_index().join(
                    other._data.to_pandas_index(), how="outer"
                )
            elif isinstance(other, Series):
                column_names_list = self._data.to_pandas_index().join(
                    other.index.to_pandas(), how="outer"
                )
            else:
                raise ValueError("other must be a DataFrame or Series.")

            sorted_dict = {key: operands[key] for key in column_names_list}
            return sorted_dict, index, can_use_self_column_name
        return operands, index, can_use_self_column_name

    @classmethod
    @_cudf_nvtx_annotate
    def from_dict(
        cls,
        data: dict,
        orient: str = "columns",
        dtype: Optional[Dtype] = None,
        columns: Optional[list] = None,
    ) -> DataFrame:
        """
        Construct DataFrame from dict of array-like or dicts.
        Creates DataFrame object from dictionary by columns or by index
        allowing dtype specification.

        Parameters
        ----------
        data : dict
            Of the form {field : array-like} or {field : dict}.
        orient : {'columns', 'index', 'tight'}, default 'columns'
            The "orientation" of the data. If the keys of the passed dict
            should be the columns of the resulting DataFrame, pass 'columns'
            (default). Otherwise if the keys should be rows, pass 'index'.
            If 'tight', assume a dict with keys ['index', 'columns', 'data',
            'index_names', 'column_names'].
        dtype : dtype, default None
            Data type to force, otherwise infer.
        columns : list, default None
            Column labels to use when ``orient='index'``. Raises a ``ValueError``
            if used with ``orient='columns'`` or ``orient='tight'``.

        Returns
        -------
        DataFrame

        See Also
        --------
        DataFrame.from_records : DataFrame from structured ndarray, sequence
            of tuples or dicts, or DataFrame.
        DataFrame : DataFrame object creation using constructor.
        DataFrame.to_dict : Convert the DataFrame to a dictionary.

        Examples
        --------
        By default the keys of the dict become the DataFrame columns:

        >>> import cudf
        >>> data = {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']}
        >>> cudf.DataFrame.from_dict(data)
           col_1 col_2
        0      3     a
        1      2     b
        2      1     c
        3      0     d

        Specify ``orient='index'`` to create the DataFrame using dictionary
        keys as rows:

        >>> data = {'row_1': [3, 2, 1, 0], 'row_2': [10, 11, 12, 13]}
        >>> cudf.DataFrame.from_dict(data, orient='index')
                0   1   2   3
        row_1   3   2   1   0
        row_2  10  11  12  13

        When using the 'index' orientation, the column names can be
        specified manually:

        >>> cudf.DataFrame.from_dict(data, orient='index',
        ...                          columns=['A', 'B', 'C', 'D'])
                A   B   C   D
        row_1   3   2   1   0
        row_2  10  11  12  13

        Specify ``orient='tight'`` to create the DataFrame using a 'tight'
        format:

        >>> data = {'index': [('a', 'b'), ('a', 'c')],
        ...         'columns': [('x', 1), ('y', 2)],
        ...         'data': [[1, 3], [2, 4]],
        ...         'index_names': ['n1', 'n2'],
        ...         'column_names': ['z1', 'z2']}
        >>> cudf.DataFrame.from_dict(data, orient='tight')
        z1     x  y
        z2     1  2
        n1 n2
        a  b   1  3
           c   2  4
        """  # noqa: E501

        orient = orient.lower()
        if orient == "index":
            if len(data) > 0 and isinstance(
                next(iter(data.values())), (cudf.Series, cupy.ndarray)
            ):
                result = cls(data).T
                result.columns = (
                    columns
                    if columns is not None
                    else range(len(result._data))
                )
                if dtype is not None:
                    result = result.astype(dtype)
                return result
            else:
                return cls.from_pandas(
                    pd.DataFrame.from_dict(
                        data=data,
                        orient=orient,
                        dtype=dtype,
                        columns=columns,
                    )
                )
        elif orient == "columns":
            if columns is not None:
                raise ValueError(
                    "Cannot use columns parameter with orient='columns'"
                )
            return cls(data, columns=None, dtype=dtype)
        elif orient == "tight":
            if columns is not None:
                raise ValueError(
                    "Cannot use columns parameter with orient='right'"
                )

            index = _from_dict_create_index(
                data["index"], data["index_names"], cudf
            )
            columns = _from_dict_create_index(
                data["columns"], data["column_names"], pd
            )
            return cls(data["data"], index=index, columns=columns, dtype=dtype)
        else:
            raise ValueError(
                "Expected 'index', 'columns' or 'tight' for orient "
                f"parameter. Got '{orient}' instead"
            )

    @_cudf_nvtx_annotate
    def to_dict(
        self,
        orient: str = "dict",
        into: type[dict] = dict,
    ) -> dict | list[dict]:
        """
        Convert the DataFrame to a dictionary.

        The type of the key-value pairs can be customized with the parameters
        (see below).

        Parameters
        ----------
        orient : str {'dict', 'list', 'series', 'split', 'tight', 'records', 'index'}
            Determines the type of the values of the dictionary.

            - 'dict' (default) : dict like {column -> {index -> value}}
            - 'list' : dict like {column -> [values]}
            - 'series' : dict like {column -> Series(values)}
            - 'split' : dict like
              {'index' -> [index], 'columns' -> [columns], 'data' -> [values]}
            - 'tight' : dict like
              {'index' -> [index], 'columns' -> [columns], 'data' -> [values],
              'index_names' -> [index.names], 'column_names' -> [column.names]}
            - 'records' : list like
              [{column -> value}, ... , {column -> value}]
            - 'index' : dict like {index -> {column -> value}}

            Abbreviations are allowed. `s` indicates `series` and `sp`
            indicates `split`.

        into : class, default dict
            The collections.abc.Mapping subclass used for all Mappings
            in the return value.  Can be the actual class or an empty
            instance of the mapping type you want.  If you want a
            collections.defaultdict, you must pass it initialized.

        Returns
        -------
        dict, list or collections.abc.Mapping
            Return a collections.abc.Mapping object representing the DataFrame.
            The resulting transformation depends on the `orient` parameter.

        See Also
        --------
        DataFrame.from_dict: Create a DataFrame from a dictionary.
        DataFrame.to_json: Convert a DataFrame to JSON format.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'col1': [1, 2],
        ...                      'col2': [0.5, 0.75]},
        ...                     index=['row1', 'row2'])
        >>> df
              col1  col2
        row1     1  0.50
        row2     2  0.75
        >>> df.to_dict()
        {'col1': {'row1': 1, 'row2': 2}, 'col2': {'row1': 0.5, 'row2': 0.75}}

        You can specify the return orientation.

        >>> df.to_dict('series')
        {'col1': row1    1
                 row2    2
        Name: col1, dtype: int64,
        'col2': row1    0.50
                row2    0.75
        Name: col2, dtype: float64}

        >>> df.to_dict('split')
        {'index': ['row1', 'row2'], 'columns': ['col1', 'col2'],
         'data': [[1, 0.5], [2, 0.75]]}

        >>> df.to_dict('records')
        [{'col1': 1, 'col2': 0.5}, {'col1': 2, 'col2': 0.75}]

        >>> df.to_dict('index')
        {'row1': {'col1': 1, 'col2': 0.5}, 'row2': {'col1': 2, 'col2': 0.75}}

        >>> df.to_dict('tight')
        {'index': ['row1', 'row2'], 'columns': ['col1', 'col2'],
         'data': [[1, 0.5], [2, 0.75]], 'index_names': [None], 'column_names': [None]}

        You can also specify the mapping type.

        >>> from collections import OrderedDict, defaultdict
        >>> df.to_dict(into=OrderedDict)
        OrderedDict([('col1', OrderedDict([('row1', 1), ('row2', 2)])),
                     ('col2', OrderedDict([('row1', 0.5), ('row2', 0.75)]))])

        If you want a `defaultdict`, you need to initialize it:

        >>> dd = defaultdict(list)
        >>> df.to_dict('records', into=dd)
        [defaultdict(<class 'list'>, {'col1': 1, 'col2': 0.5}),
         defaultdict(<class 'list'>, {'col1': 2, 'col2': 0.75})]
        """  # noqa: E501
        orient = orient.lower()

        if orient == "series":
            # Special case needed to avoid converting
            # cudf.Series objects into pd.Series
            if not inspect.isclass(into):
                cons = type(into)  # type: ignore[assignment]
                if isinstance(into, defaultdict):
                    cons = functools.partial(cons, into.default_factory)
            elif issubclass(into, abc.Mapping):
                cons = into  # type: ignore[assignment]
                if issubclass(into, defaultdict):
                    raise TypeError(
                        "to_dict() only accepts initialized defaultdicts"
                    )
            else:
                raise TypeError(f"unsupported type: {into}")
            return cons(self.items())  # type: ignore[misc]

        return self.to_pandas().to_dict(orient=orient, into=into)

    @_cudf_nvtx_annotate
    def scatter_by_map(
        self, map_index, map_size=None, keep_index=True, debug: bool = False
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

        Raises
        ------
        ValueError
            If the map_index has invalid entries (not all in [0,
            num_partitions)).
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
            cat_index = cast(
                cudf.core.column.CategoricalColumn,
                map_index.as_categorical_column("category"),
            )
            map_index = cat_index.codes
            warnings.warn(
                "Using StringColumn for map_index in scatter_by_map. "
                "Use an integer array/column for better performance."
            )
        elif isinstance(map_index, cudf.core.column.CategoricalColumn):
            map_index = map_index.codes
            warnings.warn(
                "Using CategoricalColumn for map_index in scatter_by_map. "
                "Use an integer array/column for better performance."
            )

        if debug and map_size is not None:
            count = map_index.distinct_count()
            if map_size < count:
                raise ValueError(
                    f"ERROR: map_size must be >= {count} (got {map_size})."
                )

        partitioned_columns, output_offsets = libcudf.partitioning.partition(
            [*(self._index._columns if keep_index else ()), *self._columns],
            map_index,
            map_size,
        )
        partitioned = self._from_columns_like_self(
            partitioned_columns,
            column_names=self._column_names,
            index_names=list(self._index_names) if keep_index else None,
        )

        # due to the split limitation mentioned
        # here: https://github.com/rapidsai/cudf/issues/4607
        # we need to remove first & last elements in offsets.
        # TODO: Remove this after the above issue is fixed.
        output_offsets = output_offsets[1:-1]

        result = partitioned._split(output_offsets, keep_index=keep_index)

        if map_size:
            result += [
                self._empty_like(keep_index)
                for _ in range(map_size - len(result))
            ]

        return result

    @_cudf_nvtx_annotate
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
        ------
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

        self_cols = self._data.to_pandas_index()
        if not self_cols.equals(other._data.to_pandas_index()):
            other = other.reindex(self_cols, axis=1)
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

    @_cudf_nvtx_annotate
    def __iter__(self):
        return iter(self._column_names)

    @_cudf_nvtx_annotate
    def __contains__(self, item):
        # This must check against containment in the pandas Index and not
        # self._column_names to handle NA, None, nan, etc. correctly.
        return item in self._data.to_pandas_index()

    @_cudf_nvtx_annotate
    def items(self):
        """Iterate over column names and series pairs"""
        for k in self:
            yield (k, self[k])

    @_cudf_nvtx_annotate
    def equals(self, other):
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
    @_external_only_api(
        "Use _column_names instead, or _data.to_pandas_index() if a pandas "
        "index is absolutely necessary. For checking if the columns are a "
        "MultiIndex, use _data.multiindex."
    )
    @_cudf_nvtx_annotate
    def columns(self):
        """Returns a tuple of columns"""
        return self._data.to_pandas_index()

    @columns.setter  # type: ignore
    @_cudf_nvtx_annotate
    def columns(self, columns):
        multiindex = False
        rangeindex = False
        label_dtype = None
        level_names = None
        if isinstance(columns, (pd.MultiIndex, cudf.MultiIndex)):
            multiindex = True
            if isinstance(columns, cudf.MultiIndex):
                pd_columns = columns.to_pandas()
            else:
                pd_columns = columns
            if pd_columns.nunique(dropna=False) != len(pd_columns):
                raise ValueError("Duplicate column names are not allowed")
            level_names = list(pd_columns.names)
        elif isinstance(columns, (cudf.BaseIndex, ColumnBase, Series)):
            level_names = (getattr(columns, "name", None),)
            rangeindex = isinstance(columns, cudf.RangeIndex)
            columns = as_column(columns)
            if columns.distinct_count(dropna=False) != len(columns):
                raise ValueError("Duplicate column names are not allowed")
            pd_columns = pd.Index(columns.to_pandas())
            label_dtype = pd_columns.dtype
        else:
            pd_columns = pd.Index(columns)
            if pd_columns.nunique(dropna=False) != len(pd_columns):
                raise ValueError("Duplicate column names are not allowed")
            rangeindex = isinstance(pd_columns, pd.RangeIndex)
            level_names = (pd_columns.name,)
            label_dtype = pd_columns.dtype

        if len(pd_columns) != len(self._data.names):
            raise ValueError(
                f"Length mismatch: expected {len(self._data.names)} elements, "
                f"got {len(pd_columns)} elements"
            )

        self._data = ColumnAccessor(
            data=dict(zip(pd_columns, self._data.columns)),
            multiindex=multiindex,
            level_names=level_names,
            label_dtype=label_dtype,
            rangeindex=rangeindex,
            verify=False,
        )

    def _set_columns_like(self, other: ColumnAccessor) -> None:
        """
        Modify self with the column properties of other.

        * Whether .columns is a MultiIndex/RangeIndex
        * The possible .columns.dtype
        * The .columns.names/name (depending on if it's a MultiIndex)
        """
        if len(self._data.names) != len(other.names):
            raise ValueError(
                f"Length mismatch: expected {len(other)} elements, "
                f"got {len(self)} elements"
            )
        self._data = ColumnAccessor(
            data=dict(zip(other.names, self._data.columns)),
            multiindex=other.multiindex,
            level_names=other.level_names,
            label_dtype=other.label_dtype,
            verify=False,
        )

    @_cudf_nvtx_annotate
    def reindex(
        self,
        labels=None,
        index=None,
        columns=None,
        axis=None,
        method=None,
        copy=True,
        level=None,
        fill_value=NA,
        limit=None,
        tolerance=None,
    ):
        """
        Conform DataFrame to new index. Places NA/NaN in locations
        having no value in the previous index. A new object is produced
        unless the new index is equivalent to the current one and copy=False.

        Parameters
        ----------
        labels : Index, Series-convertible, optional, default None
            New labels / index to conform the axis specified by ``axis`` to.
        index : Index, Series-convertible, optional, default None
            The index labels specifying the index to conform to.
        columns : array-like, optional, default None
            The column labels specifying the columns to conform to.
        axis : Axis to target.
            Can be either the axis name
            (``index``, ``columns``) or number (0, 1).
        method : Not supported
        copy : boolean, default True
            Return a new object, even if the passed indexes are the same.
        level : Not supported
        fill_value : Value to use for missing values.
            Defaults to ``NA``, but can be any "compatible" value.
        limit : Not supported
        tolerance : Not supported

        Returns
        -------
        DataFrame with changed index.

        Examples
        --------
        ``DataFrame.reindex`` supports two calling conventions
        * ``(index=index_labels, columns=column_labels, ...)``
        * ``(labels, axis={'index', 'columns'}, ...)``
        We _highly_ recommend using keyword arguments to clarify your intent.

        Create a dataframe with some fictional data.

        >>> index = ['Firefox', 'Chrome', 'Safari', 'IE10', 'Konqueror']
        >>> df = cudf.DataFrame({'http_status': [200, 200, 404, 404, 301],
        ...                    'response_time': [0.04, 0.02, 0.07, 0.08, 1.0]},
        ...                      index=index)
        >>> df
                http_status  response_time
        Firefox            200           0.04
        Chrome             200           0.02
        Safari             404           0.07
        IE10               404           0.08
        Konqueror          301           1.00
        >>> new_index = ['Safari', 'Iceweasel', 'Comodo Dragon', 'IE10',
        ...              'Chrome']
        >>> df.reindex(new_index)
                    http_status response_time
        Safari                404          0.07
        Iceweasel            <NA>          <NA>
        Comodo Dragon        <NA>          <NA>
        IE10                  404          0.08
        Chrome                200          0.02

        .. pandas-compat::
            **DataFrame.reindex**

            Note: One difference from Pandas is that ``NA`` is used for rows
            that do not match, rather than ``NaN``. One side effect of this is
            that the column ``http_status`` retains an integer dtype in cuDF
            where it is cast to float in Pandas.

        We can fill in the missing values by
        passing a value to the keyword ``fill_value``.

        >>> df.reindex(new_index, fill_value=0)
                    http_status  response_time
        Safari                 404           0.07
        Iceweasel                0           0.00
        Comodo Dragon            0           0.00
        IE10                   404           0.08
        Chrome                 200           0.02

        We can also reindex the columns.

        >>> df.reindex(columns=['http_status', 'user_agent'])
                http_status user_agent
        Firefox            200       <NA>
        Chrome             200       <NA>
        Safari             404       <NA>
        IE10               404       <NA>
        Konqueror          301       <NA>

        Or we can use "axis-style" keyword arguments

        >>> df.reindex(columns=['http_status', 'user_agent'])
                http_status user_agent
        Firefox            200       <NA>
        Chrome             200       <NA>
        Safari             404       <NA>
        IE10               404       <NA>
        Konqueror          301       <NA>
        """

        if labels is None and index is None and columns is None:
            return self.copy(deep=copy)

        # pandas simply ignores the labels keyword if it is provided in
        # addition to index and columns, but it prohibits the axis arg.
        if (index is not None or columns is not None) and axis is not None:
            raise TypeError(
                "Cannot specify both 'axis' and any of 'index' or 'columns'."
            )

        axis = 0 if axis is None else self._get_axis_from_axis_arg(axis)
        if axis == 0:
            if index is None:
                index = labels
        else:
            if columns is None:
                columns = labels
        if columns is None:
            df = self
        else:
            columns = as_index(columns)
            intersection = self._data.to_pandas_index().intersection(
                columns.to_pandas()
            )
            df = self.loc[:, intersection]

        return df._reindex(
            column_names=columns,
            dtypes=self._dtypes,
            deep=copy,
            index=index,
            inplace=False,
            fill_value=fill_value,
        )

    @_cudf_nvtx_annotate
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

        Set the index to become the 'b' column:

        >>> df.set_index('b')
           a    c
        b
        a  1  1.0
        b  2  2.0
        c  3  3.0
        d  4  4.0
        e  5  5.0

        Create a MultiIndex using columns 'a' and 'b':

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
                if col in self._column_names:
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
        elif (
            len(columns_to_add) == 1
            and len(keys) == 1
            and not isinstance(keys[0], (cudf.MultiIndex, pd.MultiIndex))
        ):
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

    @_cudf_nvtx_annotate
    def where(self, cond, other=None, inplace=False):
        from cudf.core._internals.where import (
            _check_and_cast_columns_with_other,
            _make_categorical_like,
        )

        # First process the condition.
        if isinstance(cond, Series):
            cond = self._from_data(
                self._data._from_columns_like_self(
                    itertools.repeat(cond._column, len(self._column_names)),
                    verify=False,
                )
            )
        elif hasattr(cond, "__cuda_array_interface__"):
            cond = DataFrame(
                cond, columns=self._column_names, index=self.index
            )
        elif (
            hasattr(cond, "__array_interface__")
            and cond.__array_interface__["shape"] != self.shape
        ):
            raise ValueError("conditional must be same shape as self")
        elif not isinstance(cond, DataFrame):
            cond = cudf.DataFrame(cond)

        if set(self._column_names).intersection(set(cond._column_names)):
            if not self.index.equals(cond.index):
                cond = cond.reindex(self.index)
        else:
            if cond.shape != self.shape:
                raise ValueError(
                    "Array conditional must be same shape as self"
                )
            # Setting `self` column names to `cond` as it has no column names.
            cond._set_columns_like(self._data)

        # If other was provided, process that next.
        if isinstance(other, DataFrame):
            other_cols = [other._data[col] for col in self._column_names]
        elif cudf.api.types.is_scalar(other):
            other_cols = [other] * len(self._column_names)
        elif isinstance(other, cudf.Series):
            other_cols = other.to_pandas()
        else:
            other_cols = other

        if len(self._columns) != len(other_cols):
            raise ValueError(
                """Replacement list length or number of data columns
                should be equal to number of columns of self"""
            )

        out = []
        for (name, col), other_col in zip(self._data.items(), other_cols):
            col, other_col = _check_and_cast_columns_with_other(
                source_col=col,
                other=other_col,
                inplace=inplace,
            )

            if cond_col := cond._data.get(name):
                result = cudf._lib.copying.copy_if_else(
                    col, other_col, cond_col
                )

                out.append(_make_categorical_like(result, self._data[name]))
            else:
                out_mask = cudf._lib.null_mask.create_null_mask(
                    len(col),
                    state=cudf._lib.null_mask.MaskState.ALL_NULL,
                )
                out.append(col.set_mask(out_mask))

        return self._mimic_inplace(
            self._from_data_like_self(self._data._from_columns_like_self(out)),
            inplace=inplace,
        )

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

    @_cudf_nvtx_annotate
    def insert(self, loc, name, value, nan_as_null=no_default):
        """Add a column to DataFrame at the index specified by loc.

        Parameters
        ----------
        loc : int
            location to insert by index, cannot be greater then num columns + 1
        name : number or string
            name or label of column to be inserted
        value : Series or array-like
        nan_as_null : bool, Default None
            If ``None``/``True``, converts ``np.nan`` values to
            ``null`` values.
            If ``False``, leaves ``np.nan`` values as is.
        """
        if nan_as_null is no_default:
            nan_as_null = not cudf.get_option("mode.pandas_compatible")
        return self._insert(
            loc=loc,
            name=name,
            value=value,
            nan_as_null=nan_as_null,
            ignore_index=False,
        )

    @_cudf_nvtx_annotate
    def _insert(self, loc, name, value, nan_as_null=None, ignore_index=True):
        """
        Same as `insert`, with additional `ignore_index` param.

        ignore_index : bool, default True
            If True, there will be no index equality check & reindexing
            happening.
            If False, a reindexing operation is performed if
            `value.index` is not equal to `self.index`.
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

        # TODO: This check is currently necessary because
        # _is_scalar_or_zero_d_array below will treat a length 1 pd.Categorical
        # as a scalar and attempt to use column.full, which can't handle it.
        # Maybe _is_scalar_or_zero_d_array should be changed, or maybe we just
        # shouldn't support pd.Categorical at all, but those changes will at
        # least require a deprecation cycle because we currently support
        # inserting a pd.Categorical.
        if isinstance(value, pd.Categorical):
            value = cudf.core.column.categorical.pandas_categorical_as_column(
                value
            )

        if _is_scalar_or_zero_d_array(value):
            dtype = None
            if isinstance(value, (np.ndarray, cupy.ndarray)):
                dtype = value.dtype
                value = value.item()
            if libcudf.scalar._is_null_host_scalar(value):
                dtype = "str"
            value = as_column(
                value,
                length=len(self),
                dtype=dtype,
            )

        if len(self) == 0:
            if isinstance(value, (pd.Series, Series)):
                if not ignore_index:
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
            value = Series(value, nan_as_null=nan_as_null)
            if not ignore_index:
                value = value._align_to_index(
                    self._index, how="right", sort=False
                )

        value = column.as_column(value, nan_as_null=nan_as_null)

        self._data.insert(name, value, loc=loc)

    @property  # type:ignore
    @_cudf_nvtx_annotate
    def axes(self):
        """
        Return a list representing the axes of the DataFrame.

        DataFrame.axes returns a list of two elements:
        element zero is the row index and element one is the columns.

        Examples
        --------
        >>> import cudf
        >>> cdf1 = cudf.DataFrame()
        >>> cdf1["key"] = [0,0,1,1]
        >>> cdf1["k2"] = [1,2,2,3]
        >>> cdf1["val"] = [1,2,3,4]
        >>> cdf1["temp"] = [-1,2,2,3]
        >>> cdf1.axes
        [RangeIndex(start=0, stop=4, step=1),
            Index(['key', 'k2', 'val', 'temp'], dtype='object')]

        """
        return [self._index, self._data.to_pandas_index()]

    def diff(self, periods=1, axis=0):
        """
        First discrete difference of element.

        Calculates the difference of a DataFrame element compared with another
        element in the DataFrame (default is element in previous row).

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for calculating difference,
            accepts negative values.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Take difference over rows (0) or columns (1).
            Only row-wise (0) shift is supported.

        Returns
        -------
        DataFrame
            First differences of the DataFrame.

        Examples
        --------
        >>> import cudf
        >>> gdf = cudf.DataFrame({'a': [1, 2, 3, 4, 5, 6],
        ...                       'b': [1, 1, 2, 3, 5, 8],
        ...                       'c': [1, 4, 9, 16, 25, 36]})
        >>> gdf
           a  b   c
        0  1  1   1
        1  2  1   4
        2  3  2   9
        3  4  3  16
        4  5  5  25
        5  6  8  36
        >>> gdf.diff(periods=2)
              a     b     c
        0  <NA>  <NA>  <NA>
        1  <NA>  <NA>  <NA>
        2     2     1     8
        3     2     2    12
        4     2     3    16
        5     2     5    20

        .. pandas-compat::
            **DataFrame.diff**

            Diff currently only supports numeric dtype columns.
        """
        if not isinstance(periods, int):
            if not (isinstance(periods, float) and periods.is_integer()):
                raise ValueError("periods must be an integer")
            periods = int(periods)

        axis = self._get_axis_from_axis_arg(axis)
        if axis != 0:
            raise NotImplementedError("Only axis=0 is supported.")

        if abs(periods) > len(self):
            df = cudf.DataFrame._from_data(
                {
                    name: column_empty(len(self), dtype=dtype, masked=True)
                    for name, dtype in zip(self._column_names, self.dtypes)
                }
            )
            return df

        return self - self.shift(periods=periods)

    @_cudf_nvtx_annotate
    def drop_duplicates(
        self,
        subset=None,
        keep="first",
        inplace=False,
        ignore_index=False,
    ):
        """
        Return DataFrame with duplicate rows removed.

        Considering certain columns is optional. Indexes, including time
        indexes are ignored.

        Parameters
        ----------
        subset : column label or sequence of labels, optional
            Only consider certain columns for identifying duplicates, by
            default use all of the columns.
        keep : {'first', 'last', ``False``}, default 'first'
            Determines which duplicates (if any) to keep.
            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.
        inplace : bool, default ``False``
            Whether to drop duplicates in place or to return a copy.
        ignore_index : bool, default ``False``
            If True, the resulting axis will be labeled 0, 1, ..., n - 1.

        Returns
        -------
        DataFrame or None
            DataFrame with duplicates removed or None if ``inplace=True``.

        See Also
        --------
        DataFrame.value_counts: Count unique combinations of columns.

        Examples
        --------
        Consider a dataset containing ramen ratings.

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

        By default, it removes duplicate rows based on all columns.

        >>> df.drop_duplicates()
             brand style  rating
        0  Yum Yum   cup     4.0
        2  Indomie   cup     3.5
        3  Indomie  pack    15.0
        4  Indomie  pack     5.0

        To remove duplicates on specific column(s), use ``subset``.

        >>> df.drop_duplicates(subset=['brand'])
             brand style  rating
        0  Yum Yum   cup     4.0
        2  Indomie   cup     3.5

        To remove duplicates and keep last occurrences, use ``keep``.

        >>> df.drop_duplicates(subset=['brand', 'style'], keep='last')
             brand style  rating
        1  Yum Yum   cup     4.0
        2  Indomie   cup     3.5
        4  Indomie  pack     5.0
        """  # noqa: E501
        outdf = super().drop_duplicates(
            subset=subset,
            keep=keep,
            ignore_index=ignore_index,
        )

        return self._mimic_inplace(outdf, inplace=inplace)

    @_cudf_nvtx_annotate
    def pop(self, item):
        """Return a column and drop it from the DataFrame."""
        popped = self[item]
        del self[item]
        return popped

    @_cudf_nvtx_annotate
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
        a dict / Series will be left as-is. Extra labels listed don't throw an
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

        .. pandas-compat::
            **DataFrame.rename**

            * Not Supporting: level

            Rename will not overwrite column names. If a list with
            duplicates is passed, column names will be postfixed
            with a number.
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
                any(isinstance(item, str) for item in index.values())
                and self.index.dtype != "object"
            ):
                raise NotImplementedError(
                    "Implicit conversion of index to "
                    "mixed type is not yet supported."
                )

            if level is not None and isinstance(self.index, MultiIndex):
                level = self.index._get_level_label(level)
                out_index = self.index.copy(deep=copy)
                level_values = out_index.get_level_values(level)
                level_values.to_frame().replace(
                    to_replace=list(index.keys()),
                    value=list(index.values()),
                    inplace=True,
                )
                out_index._data[level] = column.as_column(level_values)
                out_index._compute_levels_and_codes()
                out = DataFrame(index=out_index)
            else:
                to_replace = list(index.keys())
                vals = list(index.values())
                is_all_na = vals.count(None) == len(vals)

                try:
                    index_data = {
                        name: col.find_and_replace(to_replace, vals, is_all_na)
                        for name, col in self.index._data.items()
                    }
                except OverflowError:
                    index_data = self.index._data.copy(deep=True)

                out = DataFrame(index=_index_from_data(index_data))
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

    @_cudf_nvtx_annotate
    def add_prefix(self, prefix):
        out = self.copy(deep=True)
        out.columns = [
            prefix + col_name for col_name in list(self._data.keys())
        ]
        return out

    @_cudf_nvtx_annotate
    def add_suffix(self, suffix):
        out = self.copy(deep=True)
        out.columns = [
            col_name + suffix for col_name in list(self._data.keys())
        ]
        return out

    @_cudf_nvtx_annotate
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

        .. pandas-compat::
            **DataFrame.agg**

            * Not supporting: ``axis``, ``*args``, ``**kwargs``

        """
        dtypes = [self[col].dtype for col in self._column_names]
        common_dtype = find_common_type(dtypes)
        if not is_bool_dtype(common_dtype) and any(
            is_bool_dtype(dtype) for dtype in dtypes
        ):
            raise MixedTypeError("Cannot create a column with mixed types")

        if any(is_string_dtype(dt) for dt in dtypes):
            raise NotImplementedError(
                "DataFrame.agg() is not supported for "
                "frames containing string columns"
            )

        if axis == 0 or axis is not None:
            raise NotImplementedError("axis not implemented yet")

        if isinstance(aggs, abc.Iterable) and not isinstance(
            aggs, (str, dict)
        ):
            result = DataFrame()
            # TODO : Allow simultaneous pass for multi-aggregation as
            # a future optimization
            for agg in aggs:
                result[agg] = getattr(self, agg)()
            return result.T.sort_index(axis=1, ascending=True)

        elif isinstance(aggs, str):
            if not hasattr(self, aggs):
                raise AttributeError(
                    f"{aggs} is not a valid function for "
                    f"'DataFrame' object"
                )
            result = DataFrame()
            result[aggs] = getattr(self, aggs)()
            result = result.iloc[:, 0]
            result.name = None
            return result

        elif isinstance(aggs, dict):
            cols = aggs.keys()
            if any(callable(val) for val in aggs.values()):
                raise NotImplementedError(
                    "callable parameter is not implemented yet"
                )
            elif all(isinstance(val, str) for val in aggs.values()):
                res = {}
                for key, value in aggs.items():
                    col = self[key]
                    if not hasattr(col, value):
                        raise AttributeError(
                            f"{value} is not a valid function for "
                            f"'Series' object"
                        )
                    res[key] = getattr(col, value)()
                result = cudf.Series(list(res.values()), index=res.keys())
            elif all(isinstance(val, abc.Iterable) for val in aggs.values()):
                idxs = set()
                for val in aggs.values():
                    if isinstance(val, str):
                        idxs.add(val)
                    elif isinstance(val, abc.Iterable):
                        idxs.update(val)
                idxs = sorted(list(idxs))
                for agg in idxs:
                    if agg is callable:
                        raise NotImplementedError(
                            "callable parameter is not implemented yet"
                        )
                result = DataFrame(index=idxs, columns=cols)
                for key in aggs.keys():
                    col = self[key]
                    col_empty = column_empty(
                        len(idxs), dtype=col.dtype, masked=True
                    )
                    ans = cudf.Series(data=col_empty, index=idxs)
                    if isinstance(aggs.get(key), abc.Iterable):
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

    @_cudf_nvtx_annotate
    def nlargest(self, n, columns, keep="first"):
        """Return the first *n* rows ordered by *columns* in descending order.

        Return the first *n* rows with the largest values in *columns*, in
        descending order. The columns that are not specified are returned as
        well, but not used for ordering.

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

        .. pandas-compat::
            **DataFrame.nlargest**

            - Only a single column is supported in *columns*
        """
        return self._n_largest_or_smallest(True, n, columns, keep)

    def nsmallest(self, n, columns, keep="first"):
        """Return the first *n* rows ordered by *columns* in ascending order.

        Return the first *n* rows with the smallest values in *columns*, in
        ascending order. The columns that are not specified are returned as
        well, but not used for ordering.

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

        .. pandas-compat::
            **DataFrame.nsmallest**

            - Only a single column is supported in *columns*
        """
        return self._n_largest_or_smallest(False, n, columns, keep)

    @_cudf_nvtx_annotate
    def swaplevel(self, i=-2, j=-1, axis=0):
        """
        Swap level i with level j.
        Calling this method does not change the ordering of the values.

        Parameters
        ----------
        i : int or str, default -2
            First level of index to be swapped.
        j : int or str, default -1
            Second level of index to be swapped.
        axis : The axis to swap levels on.
            0 or 'index' for row-wise, 1 or 'columns' for column-wise.

        Examples
        --------
        >>> import cudf
        >>> midx = cudf.MultiIndex(levels=[['llama', 'cow', 'falcon'],
        ...   ['speed', 'weight', 'length'],['first','second']],
        ...   codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2],
        ...             [0, 0, 0, 0, 0, 0, 1, 1, 1]])
        >>> cdf = cudf.DataFrame(index=midx, columns=['big', 'small'],
        ...  data=[[45, 30], [200, 100], [1.5, 1], [30, 20],
        ...         [250, 150], [1.5, 0.8], [320, 250], [1, 0.8], [0.3, 0.2]])

        >>> cdf
                                     big  small
             llama  speed  first    45.0   30.0
                    weight first   200.0  100.0
                    length first     1.5    1.0
             cow    speed  first    30.0   20.0
                    weight first   250.0  150.0
                    length first     1.5    0.8
             falcon speed  second  320.0  250.0
                    weight second    1.0    0.8
                    length second    0.3    0.2

        >>> cdf.swaplevel()
                                     big  small
             llama  first  speed    45.0   30.0
                           weight  200.0  100.0
                           length    1.5    1.0
             cow    first  speed    30.0   20.0
                           weight  250.0  150.0
                           length    1.5    0.8
             falcon second speed   320.0  250.0
                           weight    1.0    0.8
                           length    0.3    0.2
        """
        result = self.copy()

        # To get axis number
        axis = self._get_axis_from_axis_arg(axis)

        if axis == 0:
            if not isinstance(result.index, MultiIndex):
                raise TypeError("Can only swap levels on a hierarchical axis.")
            result.index = result.index.swaplevel(i, j)
        else:
            if not result._data.multiindex:
                raise TypeError("Can only swap levels on a hierarchical axis.")
            result._data = result._data.swaplevel(i, j)

        return result

    @_cudf_nvtx_annotate
    def transpose(self):
        """Transpose index and columns.

        Returns
        -------
        a new (ncol x nrow) dataframe. self is (nrow x ncol)

        .. pandas-compat::
            **DataFrame.transpose, DataFrame.T**

            Not supporting *copy* because default and only behavior is
            copy=True
        """
        index = self._data.to_pandas_index()
        columns = self.index.copy(deep=False)
        if self._num_columns == 0 or self._num_rows == 0:
            return DataFrame(index=index, columns=columns)

        # No column from index is transposed with libcudf.
        source_columns = [*self._columns]
        source_dtype = source_columns[0].dtype
        if isinstance(source_dtype, cudf.CategoricalDtype):
            if any(
                not isinstance(c.dtype, cudf.CategoricalDtype)
                for c in source_columns
            ):
                raise ValueError("Columns must all have the same dtype")
            cats = list(c.categories for c in source_columns)
            cats = cudf.core.column.concat_columns(cats).unique()
            source_columns = [
                col._set_categories(cats, is_unique=True).codes
                for col in source_columns
            ]

        if any(c.dtype != source_columns[0].dtype for c in source_columns):
            raise ValueError("Columns must all have the same dtype")

        result_columns = libcudf.transpose.transpose(source_columns)

        if isinstance(source_dtype, cudf.CategoricalDtype):
            result_columns = [
                codes._with_type_metadata(
                    cudf.core.dtypes.CategoricalDtype(categories=cats)
                )
                for codes in result_columns
            ]
        else:
            result_columns = [
                result_column._with_type_metadata(source_dtype)
                for result_column in result_columns
            ]

        # Set the old column names as the new index
        result = self.__class__._from_data(
            {i: col for i, col in enumerate(result_columns)},
            index=as_index(index),
        )
        # Set the old index as the new column names
        result.columns = columns
        return result

    T = property(transpose, doc=transpose.__doc__)

    @_cudf_nvtx_annotate
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

    @_cudf_nvtx_annotate
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
        how : {'left', 'outer', 'inner', 'leftsemi', 'leftanti'}, \
            default 'inner'
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

        Returns
        -------
            merged : DataFrame

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
        - For left or right joins, the result will be the left or
        right dtype respectively. This extends to semi and anti joins.
        - For outer joins, the result will be the union of categories
        from both sides.

        .. pandas-compat::
            **DataFrame.merge**

            DataFrames merges in cuDF result in non-deterministic row
            ordering.
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

        lhs, rhs = self, right
        merge_cls = Merge
        if how == "right":
            # Merge doesn't support right, so just swap
            how = "left"
            lhs, rhs = right, self
            left_on, right_on = right_on, left_on
            left_index, right_index = right_index, left_index
            suffixes = (suffixes[1], suffixes[0])
        elif how in {"leftsemi", "leftanti"}:
            merge_cls = MergeSemi

        return merge_cls(
            lhs,
            rhs,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            sort=sort,
            indicator=indicator,
            suffixes=suffixes,
        ).perform_merge()

    @_cudf_nvtx_annotate
    def join(
        self,
        other,
        on=None,
        how="left",
        lsuffix="",
        rsuffix="",
        sort=False,
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

        Returns
        -------
        joined : DataFrame

        .. pandas-compat::
            **DataFrame.join**

            - *other* must be a single DataFrame for now.
            - *on* is not supported yet due to lack of multi-index support.
        """
        if on is not None:
            raise NotImplementedError("The on parameter is not yet supported")

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

    @_cudf_nvtx_annotate
    @docutils.doc_apply(
        groupby_doc_template.format(
            ret=textwrap.dedent(
                """
                Returns
                -------
                DataFrameGroupBy
                    Returns a DataFrameGroupBy object that contains
                    information about the groups.
                """
            )
        )
    )
    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=no_default,
        group_keys=False,
        squeeze=False,
        observed=True,
        dropna=True,
    ):
        return super().groupby(
            by,
            axis,
            level,
            as_index,
            sort,
            group_keys,
            squeeze,
            observed,
            dropna,
        )

    def query(self, expr, local_dict=None):
        """
        Query with a boolean expression using Numba to compile a GPU kernel.

        See :meth:`pandas.DataFrame.query`.

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
        filtered : DataFrame

        Examples
        --------
        >>> df = cudf.DataFrame({
        ...     "a": [1, 2, 2],
        ...     "b": [3, 4, 5],
        ... })
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
        1 2018-10-08

        Using local_dict:

        >>> import numpy as np
        >>> import datetime
        >>> df = cudf.DataFrame()
        >>> data = np.array(['2018-10-07', '2018-10-08'], dtype='datetime64')
        >>> df['datetimes'] = data
        >>> search_date2 = datetime.datetime.strptime('2018-10-08', '%Y-%m-%d')
        >>> df.query('datetimes==@search_date',
        ...          local_dict={'search_date': search_date2})
           datetimes
        1 2018-10-08

        .. pandas-compat::
            **DataFrame.query**

            One difference from pandas is that ``query`` currently only
            supports numeric, datetime, timedelta, or bool dtypes.
        """
        # can't use `annotate` decorator here as we inspect the calling
        # environment.
        with annotate("DATAFRAME_QUERY", color="purple", domain="cudf_python"):
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
            return self._apply_boolean_mask(
                BooleanMask.from_column_unchecked(boolmask)
            )

    @_cudf_nvtx_annotate
    def apply(
        self, func, axis=1, raw=False, result_type=None, args=(), **kwargs
    ):
        """
        Apply a function along an axis of the DataFrame.
        ``apply`` relies on Numba to JIT compile ``func``.
        Thus the allowed operations within ``func`` are limited to `those
        supported by the CUDA Python Numba target
        <https://numba.readthedocs.io/en/stable/cuda/cudapysupported.html>`__.
        For more information, see the `cuDF guide to user defined functions
        <https://docs.rapids.ai/api/cudf/stable/user_guide/guide-to-udfs.html>`__.

        Some string functions and methods are supported. Refer to the guide
        to UDFs for details.

        Parameters
        ----------
        func : function
            Function to apply to each row.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Axis along which the function is applied.
            - 0 or 'index': apply function to each column (not yet supported).
            - 1 or 'columns': apply function to each row.
        raw: bool, default False
            Not yet supported
        result_type: {'expand', 'reduce', 'broadcast', None}, default None
            Not yet supported
        args: tuple
            Positional arguments to pass to func in addition to the dataframe.

        Examples
        --------
        Simple function of a single variable which could be NA:

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
        a null aware manner:

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

        Functions may conditionally return NA as in pandas:

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
        type, rather than object as in pandas:

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
        the data:

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

        Ops against N columns are supported generally:

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

        UDFs manipulating string data are allowed, as long as
        they neither modify strings in place nor create new strings.
        For example, the following UDF is allowed:

        >>> def f(row):
        ...     st = row['str_col']
        ...     scale = row['scale']
        ...     if len(st) == 0:
        ...             return -1
        ...     elif st.startswith('a'):
        ...             return 1 - scale
        ...     elif 'example' in st:
        ...             return 1 + scale
        ...     else:
        ...             return 42
        ...
        >>> df = cudf.DataFrame({
        ...     'str_col': ['', 'abc', 'some_example'],
        ...     'scale': [1, 2, 3]
        ... })
        >>> df.apply(f, axis=1)  # doctest: +SKIP
        0   -1
        1   -1
        2    4
        dtype: int64

        However, the following UDF is not allowed since it includes an
        operation that requires the creation of a new string: a call to the
        ``upper`` method. Methods that are not supported in this manner
        will raise an ``AttributeError``.

        >>> def f(row):
        ...     st = row['str_col'].upper()
        ...     return 'ABC' in st
        >>> df.apply(f, axis=1)  # doctest: +SKIP

        For a complete list of supported functions and methods that may be
        used to manipulate string data, see the UDF guide,
        <https://docs.rapids.ai/api/cudf/stable/user_guide/guide-to-udfs.html>
        """
        if axis != 1:
            raise ValueError(
                "DataFrame.apply currently only supports row wise ops"
            )
        if raw:
            raise ValueError("The `raw` kwarg is not yet supported.")
        if result_type is not None:
            raise ValueError("The `result_type` kwarg is not yet supported.")

        return self._apply(func, _get_row_kernel, *args, **kwargs)

    def applymap(
        self,
        func: Callable[[Any], Any],
        na_action: Union[str, None] = None,
        **kwargs,
    ) -> DataFrame:
        """
        Apply a function to a Dataframe elementwise.

        This method applies a function that accepts and returns a scalar
        to every element of a DataFrame.

        Parameters
        ----------
        func : callable
            Python function, returns a single value from a single value.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NaN values, without passing them to func.

        Returns
        -------
        DataFrame
            Transformed DataFrame.
        """
        # Do not remove until pandas 3.0 support is added.
        assert PANDAS_LT_300, "Need to drop after pandas-3.0 support is added."
        warnings.warn(
            "DataFrame.applymap has been deprecated. Use DataFrame.map "
            "instead.",
            FutureWarning,
        )
        return self.map(func=func, na_action=na_action, **kwargs)

    def map(
        self,
        func: Callable[[Any], Any],
        na_action: Union[str, None] = None,
        **kwargs,
    ) -> DataFrame:
        """
        Apply a function to a Dataframe elementwise.

        This method applies a function that accepts and returns a scalar
        to every element of a DataFrame.

        Parameters
        ----------
        func : callable
            Python function, returns a single value from a single value.
        na_action : {None, 'ignore'}, default None
            If 'ignore', propagate NaN values, without passing them to func.

        Returns
        -------
        DataFrame
            Transformed DataFrame.
        """

        if kwargs:
            raise NotImplementedError(
                "DataFrame.applymap does not yet support **kwargs."
            )

        if na_action not in {"ignore", None}:
            raise ValueError(
                f"na_action must be 'ignore' or None. Got {repr(na_action)}"
            )

        if na_action == "ignore":
            devfunc = numba.cuda.jit(device=True)(func)

            # promote to a null-ignoring function
            # this code is never run in python, it only
            # exists to provide numba with the correct
            # bytecode to generate the equivalent PTX
            # as a null-ignoring version of the function
            def _func(x):  # pragma: no cover
                if x is NA:
                    return NA
                else:
                    return devfunc(x)

        else:
            _func = func

        # TODO: naive implementation
        # this could be written as a single kernel
        result = {}
        for name, col in self._data.items():
            apply_sr = Series._from_data({None: col})
            result[name] = apply_sr.apply(_func)

        return DataFrame._from_data(result, index=self.index)

    @_cudf_nvtx_annotate
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
            if is_string_dtype(current_col_dtype) or isinstance(
                current_col_dtype, cudf.CategoricalDtype
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

    @_cudf_nvtx_annotate
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
        https://numba.readthedocs.io/en/stable/cuda/kernels.html

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

        See Also
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

    @_cudf_nvtx_annotate
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
        key_indices = [self._column_names.index(k) for k in columns]
        if keep_index:
            cols = [*self._index._columns, *self._columns]
            key_indices = [i + len(self._index._columns) for i in key_indices]
        else:
            cols = [*self._columns]

        output_columns, offsets = libcudf.hash.hash_partition(
            cols, key_indices, nparts
        )
        outdf = self._from_columns_like_self(
            output_columns,
            self._column_names,
            self._index_names if keep_index else None,
        )
        # Slice into partitions. Notice, `hash_partition` returns the start
        # offset of each partition thus we skip the first offset
        ret = outdf._split(offsets[1:], keep_index=keep_index)

        # Calling `_split()` on an empty dataframe returns an empty list
        # so we add empty partitions here
        ret += [self._empty_like(keep_index) for _ in range(nparts - len(ret))]
        return ret

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

        Pipe output of DataFrame.info to a buffer instead of sys.stdout and
        print buffer contents:

        >>> import io
        >>> buffer = io.StringIO()
        >>> df.info(buf=buffer)
        >>> print(buffer.getvalue())
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

        if len(self._data) == 0:
            lines.append(f"Empty {type(self).__name__}")
            cudf.utils.ioutils.buffer_write_lines(buf, lines)
            return

        cols = self._column_names
        col_count = len(cols)

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
            lines.append(f"Data columns (total {col_count} columns):")

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
                if col_count != len(counts):
                    raise AssertionError(
                        f"Columns must equal "
                        f"counts ({col_count} != {len(counts)})"
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

            for i, col in enumerate(self._column_names):
                dtype = self.dtypes.iloc[i]
                col = pprint_thing(col)

                line_no = _put_str(f" {i}", space_num)
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
            if col_count > 0:
                entries_summary = f", {cols[0]} to {cols[-1]}"
            else:
                entries_summary = ""
            columns_summary = f"Columns: {col_count} entries{entries_summary}"
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

    @_cudf_nvtx_annotate
    @docutils.doc_describe()
    def describe(
        self,
        percentiles=None,
        include=None,
        exclude=None,
    ):
        """{docstring}"""

        if not include and not exclude:
            default_include = [np.number, "datetime"]
            data_to_describe = self.select_dtypes(include=default_include)
            if data_to_describe._num_columns == 0:
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
            data_to_describe[col].describe(
                percentiles=percentiles,
            )
            for col in data_to_describe._column_names
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

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                res = cudf.concat(
                    [
                        series.reindex(names, copy=False)
                        for series in describe_series_list
                    ],
                    axis=1,
                    sort=False,
                )
            return res

    @_cudf_nvtx_annotate
    def to_pandas(
        self, *, nullable: bool = False, arrow_type: bool = False
    ) -> pd.DataFrame:
        """
        Convert to a Pandas DataFrame.

        Parameters
        ----------
        nullable : Boolean, Default False
            If ``nullable`` is ``True``, the resulting columns
            in the dataframe will be having a corresponding
            nullable Pandas dtype. If there is no corresponding
            nullable Pandas dtype present, the resulting dtype
            will be a regular pandas dtype.
            If ``nullable`` is ``False``,
            the resulting columns will either convert null
            values to ``np.nan`` or ``None`` depending on the dtype.
        arrow_type : bool, Default False
            Return the columns with a ``pandas.ArrowDtype``

        Returns
        -------
        out : Pandas DataFrame

        Notes
        -----
        nullable and arrow_type cannot both be set to ``True``

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

        ``nullable=True`` converts the result to pandas nullable types:

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

        ``arrow_type=True`` converts the result to ``pandas.ArrowDtype``:

        >>> df.to_pandas(arrow_type=True).dtypes
        a    int64[pyarrow]
        b     bool[pyarrow]
        dtype: object
        """
        out_index = self.index.to_pandas()
        out_data = {
            i: col.to_pandas(
                index=out_index, nullable=nullable, arrow_type=arrow_type
            )
            for i, col in enumerate(self._data.columns)
        }

        out_df = pd.DataFrame(out_data, index=out_index)
        out_df.columns = self._data.to_pandas_index()

        return out_df

    @classmethod
    @_cudf_nvtx_annotate
    def from_pandas(cls, dataframe, nan_as_null=no_default):
        """
        Convert from a Pandas DataFrame.

        Parameters
        ----------
        dataframe : Pandas DataFrame object
            A Pandas DataFrame object which has to be converted
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
        if nan_as_null is no_default:
            nan_as_null = (
                False if cudf.get_option("mode.pandas_compatible") else None
            )

        if isinstance(dataframe, pd.DataFrame):
            if not dataframe.columns.is_unique:
                raise ValueError("Duplicate column names are not allowed")

            data = {
                col_name: column.as_column(
                    col_value.array, nan_as_null=nan_as_null
                )
                for col_name, col_value in dataframe.items()
            }
            if isinstance(dataframe.index, pd.MultiIndex):
                index = cudf.MultiIndex.from_pandas(
                    dataframe.index, nan_as_null=nan_as_null
                )
            else:
                index = cudf.Index.from_pandas(
                    dataframe.index, nan_as_null=nan_as_null
                )
            df = cls._from_data(data, index)
            df._data._level_names = tuple(dataframe.columns.names)

            if isinstance(dataframe.columns, pd.RangeIndex):
                df._data.rangeindex = True
            # Set columns only if it is a MultiIndex
            elif isinstance(dataframe.columns, pd.MultiIndex):
                df.columns = dataframe.columns

            return df
        elif hasattr(dataframe, "__dataframe__"):
            # TODO: Probably should be handled in the constructor as
            # this isn't pandas specific
            return from_dataframe(dataframe, allow_copy=True)
        else:
            raise TypeError(
                f"Could not construct DataFrame from {type(dataframe)}"
            )

    @classmethod
    @_cudf_nvtx_annotate
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

        .. pandas-compat::
            **DataFrame.from_arrow**

            -   Does not support automatically setting index column(s) similar
                to how ``to_pandas`` works for PyArrow Tables.
        """
        index_col = None
        col_index_names = None
        physical_column_md = []
        if isinstance(table, pa.Table) and isinstance(
            table.schema.pandas_metadata, dict
        ):
            physical_column_md = table.schema.pandas_metadata["columns"]
            index_col = table.schema.pandas_metadata["index_columns"]
            if "column_indexes" in table.schema.pandas_metadata:
                col_index_names = []
                for col_meta in table.schema.pandas_metadata["column_indexes"]:
                    col_index_names.append(col_meta["name"])

        out = super().from_arrow(table)
        if col_index_names is not None:
            out._data._level_names = col_index_names
        if index_col:
            if isinstance(index_col[0], dict):
                range_meta = index_col[0]
                idx = cudf.RangeIndex(
                    start=range_meta["start"],
                    stop=range_meta["stop"],
                    step=range_meta["step"],
                    name=range_meta["name"],
                )
                if len(idx) == len(out):
                    # `idx` is generated from arrow `pandas_metadata`
                    # which can get out of date with many of the
                    # arrow operations. Hence verifying if the
                    # lengths match, or else don't need to set
                    # an index at all i.e., Default RangeIndex
                    # will be set.
                    # See more about the discussion here:
                    # https://github.com/apache/arrow/issues/15178
                    out = out.set_index(idx)
            else:
                out = out.set_index(index_col)

        if (
            "__index_level_0__" in out.index.names
            and len(out.index.names) == 1
        ):
            real_index_name = None
            for md in physical_column_md:
                if md["field_name"] == "__index_level_0__":
                    real_index_name = md["name"]
                    break
            out.index.name = real_index_name

        return out

    @_cudf_nvtx_annotate
    def to_arrow(self, preserve_index=None):
        """
        Convert to a PyArrow Table.

        Parameters
        ----------
        preserve_index : bool, optional
            whether index column and its meta data needs to be saved
            or not. The default of None will store the index as a
            column, except for a RangeIndex which is stored as
            metadata only. Setting preserve_index to True will force
            a RangeIndex to be materialized.

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
        ----
        a: [[1,2,3]]
        b: [[4,5,6]]
        index: [[1,2,3]]
        >>> df.to_arrow(preserve_index=False)
        pyarrow.Table
        a: int64
        b: int64
        ----
        a: [[1,2,3]]
        b: [[4,5,6]]
        """

        data = self.copy(deep=False)
        index_descr = []
        write_index = preserve_index is not False
        keep_range_index = write_index and preserve_index is None
        index = self.index
        index_levels = [self.index]
        if write_index:
            if isinstance(index, cudf.RangeIndex) and keep_range_index:
                index_descr = [
                    {
                        "kind": "range",
                        "name": index.name,
                        "start": index.start,
                        "stop": index.stop,
                        "step": index.step,
                    }
                ]
            else:
                if isinstance(index, cudf.RangeIndex):
                    index = index._as_int_index()
                    index.name = "__index_level_0__"
                if isinstance(index, MultiIndex):
                    index_descr = list(index._data.names)
                    index_levels = index.levels
                else:
                    index_descr = (
                        index.names if index.name is not None else ("index",)
                    )
                for gen_name, col_name in zip(index_descr, index._data.names):
                    data._insert(
                        data.shape[1],
                        gen_name,
                        index._data[col_name],
                    )

        out = super(DataFrame, data).to_arrow()
        metadata = pa.pandas_compat.construct_metadata(
            columns_to_convert=[self[col] for col in self._data.names],
            df=self,
            column_names=out.schema.names,
            index_levels=index_levels,
            index_descriptors=index_descr,
            preserve_index=preserve_index,
            types=out.schema.types,
        )

        return out.replace_schema_metadata(metadata)

    @_cudf_nvtx_annotate
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
    @_cudf_nvtx_annotate
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
        if isinstance(columns, pd.Index):
            df._data._level_names = tuple(columns.names)
        return df

    @classmethod
    @_cudf_nvtx_annotate
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
            names = range(num_cols)
        else:
            if len(columns) != num_cols:
                raise ValueError(
                    f"columns length expected {num_cols} but "
                    f"found {len(columns)}"
                )
            elif len(columns) != len(set(columns)):
                raise ValueError("Duplicate column names are not allowed")
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
        if isinstance(columns, pd.Index):
            df._data._level_names = tuple(columns.names)
        if isinstance(columns, (range, pd.RangeIndex, cudf.RangeIndex)):
            df._data.rangeindex = True

        if index is None:
            df._index = RangeIndex(start=0, stop=len(data))
        else:
            df._index = as_index(index)
        return df

    @_cudf_nvtx_annotate
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

    @_cudf_nvtx_annotate
    def quantile(
        self,
        q=0.5,
        axis=0,
        numeric_only=True,
        interpolation=None,
        columns=None,
        exact=True,
        method="single",
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
        interpolation : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            This parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points i and j.
            Default is ``'linear'`` for ``method="single"``, and ``'nearest'``
            for ``method="table"``.

                * linear: `i + (j - i) * fraction`, where `fraction` is the
                  fractional part of the index surrounded by `i` and `j`.
                * lower: `i`.
                * higher: `j`.
                * nearest: `i` or `j` whichever is nearest.
                * midpoint: (`i` + `j`) / 2.
        columns : list of str
            List of column names to include.
        exact : boolean
            Whether to use approximate or exact quantile algorithm.
        method : {'single', 'table'}, default `'single'`
            Whether to compute quantiles per-column ('single') or over all
            columns ('table'). When 'table', the only allowed interpolation
            methods are 'nearest', 'lower', and 'higher'.

        Returns
        -------
        Series or DataFrame
            If q is an array or numeric_only is set to False, a DataFrame
            will be returned where index is q, the columns are the columns
            of self, and the values are the quantile.

            If q is a float, a Series will be returned where the index is
            the columns of self and the values are the quantiles.

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

        .. pandas-compat::
            **DataFrame.quantile**

            One notable difference from Pandas is when DataFrame is of
            non-numeric types and result is expected to be a Series in case of
            Pandas. cuDF will return a DataFrame as it doesn't support mixed
            types under Series.
        """  # noqa: E501
        if axis not in (0, None):
            raise NotImplementedError("axis is not implemented yet")

        data_df = self
        if numeric_only:
            data_df = data_df.select_dtypes(
                include=[np.number], exclude=["datetime64", "timedelta64"]
            )

        if columns is None:
            columns = data_df._data.names

        if isinstance(q, numbers.Number):
            q_is_number = True
            qs = [float(q)]
        elif pd.api.types.is_list_like(q):
            q_is_number = False
            qs = q
        else:
            msg = "`q` must be either a single element or list"
            raise TypeError(msg)

        if method == "table":
            interpolation = interpolation or "nearest"
            result = self._quantile_table(qs, interpolation.upper())

            if q_is_number:
                result = result.transpose()
                return Series(
                    data=result._columns[0], index=result.index, name=q
                )
        else:
            # Ensure that qs is non-scalar so that we always get a column back.
            interpolation = interpolation or "linear"
            result = {}
            for k in data_df._data.names:
                if k in columns:
                    ser = data_df[k]
                    res = ser.quantile(
                        qs,
                        interpolation=interpolation,
                        exact=exact,
                        quant_index=False,
                    )._column
                    if len(res) == 0:
                        res = column.column_empty_like(
                            qs, dtype=ser.dtype, masked=True, newsize=len(qs)
                        )
                    result[k] = res
            result = DataFrame._from_data(result)

            if q_is_number and numeric_only:
                result = result.fillna(np.nan).iloc[0]
                result.index = data_df.keys()
                result.name = q
                return result

        result.index = cudf.Index(list(map(float, qs)), dtype="float64")
        return result

    @_cudf_nvtx_annotate
    def isin(self, values):
        """
        Whether each element in the DataFrame is contained in values.

        Parameters
        ----------
        values : iterable, Series, DataFrame or dict
            The result will only be true at a location if all
            the labels match. If values is a Series, that's the index.
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
        # TODO: propagate nulls through isin
        # https://github.com/rapidsai/cudf/issues/7556

        fill_value = cudf.Scalar(False)

        def make_false_column_like_self():
            return column.as_column(fill_value, length=len(self), dtype="bool")

        # Preprocess different input types into a mapping from column names to
        # a list of values to check.
        result = {}
        if isinstance(values, IndexedFrame):
            # Note: In the case where values is a Series, computing some
            # information about the values column outside the loop may result
            # in performance gains.  However, since categorical conversion
            # depends on the current column in the loop, using the correct
            # precomputed variables inside the loop requires nontrivial logic.
            # This optimization could be attempted if `isin` ever becomes a
            # bottleneck.
            if (
                isinstance(values, (Series, DataFrame))
                and not values.index.is_unique
            ):
                # if DataFrame ever supports duplicate columns
                # would need to check that here
                raise ValueError("cannot compute isin with a duplicate axis.")
            values = values.reindex(self.index)
            other_cols = (
                values._data
                if isinstance(values, DataFrame)
                else {name: values._column for name in self._data}
            )
            for col, self_col in self._data.items():
                if col in other_cols:
                    other_col = other_cols[col]
                    self_is_cat = isinstance(self_col, CategoricalColumn)
                    other_is_cat = isinstance(other_col, CategoricalColumn)

                    if self_is_cat != other_is_cat:
                        # It is valid to compare the levels of a categorical
                        # column to a non-categorical column.
                        if self_is_cat:
                            self_col = self_col._get_decategorized_column()
                        else:
                            other_col = other_col._get_decategorized_column()

                    # We use the type checks from _before_ the conversion
                    # because if only one was categorical then it's already
                    # been converted and we have to check if they're strings.
                    if self_is_cat and other_is_cat:
                        self_is_str = other_is_str = False
                    else:
                        # These checks must happen after the conversions above
                        # since numpy can't handle categorical dtypes.
                        self_is_str = is_string_dtype(self_col.dtype)
                        other_is_str = is_string_dtype(other_col.dtype)

                    if self_is_str != other_is_str:
                        # Strings can't compare to anything else.
                        result[col] = make_false_column_like_self()
                    else:
                        result[col] = (self_col == other_col).fillna(False)
                else:
                    result[col] = make_false_column_like_self()
        elif is_dict_like(values):
            for name, col in self._data.items():
                if name in values:
                    result[name] = col.isin(values[name])
                else:
                    result[name] = make_false_column_like_self()
        elif is_list_like(values):
            for name, col in self._data.items():
                result[name] = col.isin(values)
        else:
            raise TypeError(
                "only list-like or dict-like objects are "
                "allowed to be passed to DataFrame.isin(), "
                "you passed a "
                f"'{type(values).__name__}'"
            )

        # TODO: Update this logic to properly preserve MultiIndex columns.
        return DataFrame._from_data(result, self.index)

    #
    # Stats
    #
    @_cudf_nvtx_annotate
    def _prepare_for_rowwise_op(self, method, skipna, numeric_only):
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

        if numeric_only:
            filtered = self.select_dtypes(include=[np.number, np.bool_])
        else:
            filtered = self.copy(deep=False)

        is_pure_dt = all(is_datetime_dtype(dt) for dt in filtered.dtypes)

        common_dtype = find_common_type(filtered.dtypes)
        if (
            not numeric_only
            and is_string_dtype(common_dtype)
            and any(not is_string_dtype(dt) for dt in filtered.dtypes)
        ):
            raise TypeError(
                f"Cannot perform row-wise {method} across mixed-dtype columns,"
                " try type-casting all the columns to same dtype."
            )

        if not skipna and any(col.nullable for col in filtered._columns):
            mask = DataFrame(
                {
                    name: filtered._data[name]._get_mask_as_column()
                    if filtered._data[name].nullable
                    else as_column(True, length=len(filtered._data[name]))
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

    @_cudf_nvtx_annotate
    def count(self, axis=0, numeric_only=False):
        """
        Count ``non-NA`` cells for each column or row.

        The values ``None``, ``NaN``, ``NaT`` are considered ``NA``.

        Returns
        -------
        Series
            For each column/row the number of non-NA/null entries.

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

        .. pandas-compat::
            **DataFrame.count**

            Parameters currently not supported are `axis` and `numeric_only`.
        """
        axis = self._get_axis_from_axis_arg(axis)
        if axis != 0:
            raise NotImplementedError("Only axis=0 is currently supported.")
        length = len(self)
        return Series._from_data(
            {
                None: [
                    length - self._data[col].null_count
                    for col in self._data.names
                ]
            },
            as_index(self._data.names),
        )

    _SUPPORT_AXIS_LOOKUP = {
        0: 0,
        1: 1,
        "index": 0,
        "columns": 1,
    }

    @_cudf_nvtx_annotate
    def _reduce(
        self,
        op,
        axis=None,
        numeric_only=False,
        **kwargs,
    ):
        source = self

        if axis is None:
            if op in {"sum", "product", "std", "var"}:
                # Do not remove until pandas 2.0 support is added.
                warnings.warn(
                    f"In a future version, {type(self).__name__}"
                    f".{op}(axis=None) will return a scalar {op} over "
                    "the entire DataFrame. To retain the old behavior, "
                    f"use '{type(self).__name__}.{op}(axis=0)' or "
                    f"just '{type(self)}.{op}()'",
                    FutureWarning,
                )
                axis = 0
            else:
                axis = 2
        elif axis is no_default:
            axis = 0
        else:
            axis = source._get_axis_from_axis_arg(axis)

        if numeric_only:
            numeric_cols = (
                name
                for name in self._data.names
                if is_numeric_dtype(self._data[name].dtype)
            )
            source = self._get_columns_by_label(numeric_cols)
            if source.empty:
                return Series(
                    index=self._data.to_pandas_index()[:0]
                    if axis == 0
                    else source.index,
                    dtype="float64",
                )
        if axis in {0, 2}:
            if axis == 2 and op in ("kurtosis", "kurt", "skew"):
                # TODO: concat + op can probably be done in the general case
                # for axis == 2.
                # https://github.com/rapidsai/cudf/issues/14930
                return getattr(concat_columns(source._data.columns), op)(
                    **kwargs
                )
            try:
                result = [
                    getattr(source._data[col], op)(**kwargs)
                    for col in source._data.names
                ]
            except AttributeError:
                numeric_ops = (
                    "mean",
                    "min",
                    "max",
                    "sum",
                    "product",
                    "prod",
                    "std",
                    "var",
                    "kurtosis",
                    "kurt",
                    "skew",
                )

                if op in numeric_ops:
                    if numeric_only:
                        try:
                            result = [
                                getattr(source._data[col], op)(**kwargs)
                                for col in source._data.names
                            ]
                        except AttributeError:
                            raise NotImplementedError(
                                f"Not all column dtypes support op {op}"
                            )
                    elif any(
                        not is_numeric_dtype(self._data[name].dtype)
                        for name in self._data.names
                    ):
                        raise TypeError(
                            "Non numeric columns passed with "
                            "`numeric_only=False`, pass `numeric_only=True` "
                            f"to perform DataFrame.{op}"
                        )
                else:
                    raise
            if axis == 2:
                return getattr(as_column(result, nan_as_null=False), op)(
                    **kwargs
                )
            else:
                source_dtypes = [c.dtype for c in source._data.columns]
                common_dtype = find_common_type(source_dtypes)
                if (
                    is_object_dtype(common_dtype)
                    and any(
                        not is_object_dtype(dtype) for dtype in source_dtypes
                    )
                    or not is_bool_dtype(common_dtype)
                    and any(is_bool_dtype(dtype) for dtype in source_dtypes)
                ):
                    raise TypeError(
                        "Columns must all have the same dtype to "
                        f"perform {op=} with {axis=}"
                    )
                if source._data.multiindex:
                    idx = MultiIndex.from_tuples(
                        source._data.names, names=source._data.level_names
                    )
                else:
                    idx = as_index(source._data.names)
                return Series._from_data({None: as_column(result)}, idx)
        elif axis == 1:
            return source._apply_cupy_method_axis_1(op, **kwargs)
        else:
            raise ValueError(f"Invalid value of {axis=} received for {op}")

    @_cudf_nvtx_annotate
    def _scan(
        self,
        op,
        axis=None,
        *args,
        **kwargs,
    ):
        if axis is None:
            axis = 0
        axis = self._get_axis_from_axis_arg(axis)

        if axis == 0:
            return super()._scan(op, axis=axis, *args, **kwargs)
        elif axis == 1:
            return self._apply_cupy_method_axis_1(op, **kwargs)

    @_cudf_nvtx_annotate
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

        .. pandas-compat::
            **DataFrame.mode**

            ``axis`` parameter is currently not supported.
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

        with warnings.catch_warnings():
            assert (
                PANDAS_LT_300
            ), "Need to drop after pandas-3.0 support is added."
            warnings.simplefilter("ignore", FutureWarning)
            df = cudf.concat(mode_results, axis=1)

        if isinstance(df, Series):
            df = df.to_frame()

        df._set_columns_like(data_df._data)

        return df

    @_cudf_nvtx_annotate
    def all(self, axis=0, bool_only=None, skipna=True, **kwargs):
        obj = self.select_dtypes(include="bool") if bool_only else self
        return super(DataFrame, obj).all(axis, skipna, **kwargs)

    @_cudf_nvtx_annotate
    def any(self, axis=0, bool_only=None, skipna=True, **kwargs):
        obj = self.select_dtypes(include="bool") if bool_only else self
        return super(DataFrame, obj).any(axis, skipna, **kwargs)

    @_cudf_nvtx_annotate
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

        numeric_only = kwargs.pop("numeric_only", False)

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
            method, skipna, numeric_only
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
        arr = prepared.to_cupy()

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
            return Series(
                result,
                index=self.index,
                dtype=result_dtype,
            )
        else:
            result_df = DataFrame(result).set_index(self.index)
            result_df._set_columns_like(prepared._data)
            return result_df

    @_cudf_nvtx_annotate
    def _columns_view(self, columns):
        """
        Return a subset of the DataFrame's columns as a view.
        """
        return DataFrame(
            {col: self._data[col] for col in columns}, index=self.index
        )

    @_cudf_nvtx_annotate
    def select_dtypes(self, include=None, exclude=None):
        """Return a subset of the DataFrame's columns based on the column dtypes.

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
        """  # noqa: E501

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
                if i_dtype == cudf.CategoricalDtype:
                    # Matches cudf & pandas dtype objects
                    include_subtypes.add(i_dtype)
                elif inspect.isclass(dtype.type):
                    if issubclass(dtype.type, i_dtype):
                        include_subtypes.add(dtype.type)

        # exclude all subtypes
        exclude_subtypes = set()
        for dtype in self.dtypes:
            for e_dtype in exclude:
                # category handling
                if e_dtype == cudf.CategoricalDtype:
                    # Matches cudf & pandas dtype objects
                    exclude_subtypes.add(e_dtype)
                elif inspect.isclass(dtype.type):
                    if issubclass(dtype.type, e_dtype):
                        exclude_subtypes.add(dtype.type)

        include_all = {cudf_dtype_from_pydata_dtype(d) for d in self.dtypes}

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
                df._insert(len(df._data), k, col)

        return df

    @ioutils.doc_to_parquet()
    def to_parquet(
        self,
        path,
        engine="cudf",
        compression="snappy",
        index=None,
        partition_cols=None,
        partition_file_name=None,
        partition_offsets=None,
        statistics="ROWGROUP",
        metadata_file_path=None,
        int96_timestamps=False,
        row_group_size_bytes=ioutils._ROW_GROUP_SIZE_BYTES_DEFAULT,
        row_group_size_rows=None,
        max_page_size_bytes=None,
        max_page_size_rows=None,
        storage_options=None,
        return_metadata=False,
        use_dictionary=True,
        header_version="1.0",
        *args,
        **kwargs,
    ):
        """{docstring}"""
        from cudf.io import parquet

        return parquet.to_parquet(
            self,
            path=path,
            engine=engine,
            compression=compression,
            index=index,
            partition_cols=partition_cols,
            partition_file_name=partition_file_name,
            partition_offsets=partition_offsets,
            statistics=statistics,
            metadata_file_path=metadata_file_path,
            int96_timestamps=int96_timestamps,
            row_group_size_bytes=row_group_size_bytes,
            row_group_size_rows=row_group_size_rows,
            max_page_size_bytes=max_page_size_bytes,
            max_page_size_rows=max_page_size_rows,
            storage_options=storage_options,
            return_metadata=return_metadata,
            use_dictionary=use_dictionary,
            header_version=header_version,
            *args,
            **kwargs,
        )

    @ioutils.doc_to_feather()
    def to_feather(self, path, *args, **kwargs):
        """{docstring}"""
        from cudf.io import feather

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
        encoding=None,
        compression=None,
        lineterminator=None,
        chunksize=None,
        storage_options=None,
    ):
        """{docstring}"""
        from cudf.io import csv

        if lineterminator is None:
            lineterminator = os.linesep
        return csv.to_csv(
            self,
            path_or_buf=path_or_buf,
            sep=sep,
            na_rep=na_rep,
            columns=columns,
            header=header,
            index=index,
            lineterminator=lineterminator,
            chunksize=chunksize,
            encoding=encoding,
            compression=compression,
            storage_options=storage_options,
        )

    @ioutils.doc_to_orc()
    def to_orc(
        self,
        fname,
        compression="snappy",
        statistics="ROWGROUP",
        stripe_size_bytes=None,
        stripe_size_rows=None,
        row_index_stride=None,
        cols_as_map_type=None,
        storage_options=None,
        index=None,
    ):
        """{docstring}"""
        from cudf.io import orc

        return orc.to_orc(
            df=self,
            fname=fname,
            compression=compression,
            statistics=statistics,
            stripe_size_bytes=stripe_size_bytes,
            stripe_size_rows=stripe_size_rows,
            row_index_stride=row_index_stride,
            cols_as_map_type=cols_as_map_type,
            storage_options=storage_options,
            index=index,
        )

    @_cudf_nvtx_annotate
    def stack(self, level=-1, dropna=no_default, future_stack=False):
        """Stack the prescribed level(s) from columns to index

        Return a reshaped DataFrame or Series having a multi-level
        index with one or more new inner-most levels compared to
        the current DataFrame. The new inner-most levels are created
        by pivoting the columns of the current dataframe:

          - if the columns have a single level, the output is a Series;
          - if the columns have multiple levels, the new index
            level(s) is (are) taken from the prescribed level(s) and
            the output is a DataFrame.

        Parameters
        ----------
        level : int, str, list default -1
            Level(s) to stack from the column axis onto the index axis,
            defined as one index or label, or a list of indices or labels.
        dropna : bool, default True
            Whether to drop rows in the resulting Frame/Series with missing
            values. When multiple levels are specified, `dropna==False` is
            unsupported.

        Returns
        -------
        DataFrame or Series
            Stacked dataframe or series.

        See Also
        --------
        DataFrame.unstack : Unstack prescribed level(s) from index axis
             onto column axis.
        DataFrame.pivot : Reshape dataframe from long format to wide
             format.
        DataFrame.pivot_table : Create a spreadsheet-style pivot table
             as a DataFrame.

        Notes
        -----
        The function is named by analogy with a collection of books
        being reorganized from being side by side on a horizontal
        position (the columns of the dataframe) to being stacked
        vertically on top of each other (in the index of the
        dataframe).

        Examples
        --------
        **Single level columns**

        >>> df_single_level_cols = cudf.DataFrame([[0, 1], [2, 3]],
        ...                                     index=['cat', 'dog'],
        ...                                     columns=['weight', 'height'])

        Stacking a dataframe with a single level column axis returns a Series:

        >>> df_single_level_cols
             weight height
        cat       0      1
        dog       2      3
        >>> df_single_level_cols.stack()
        cat  height    1
             weight    0
        dog  height    3
             weight    2
        dtype: int64

        **Multi level columns: simple case**

        >>> import pandas as pd
        >>> multicol1 = pd.MultiIndex.from_tuples([('weight', 'kg'),
        ...                                        ('weight', 'pounds')])
        >>> df_multi_level_cols1 = cudf.DataFrame([[1, 2], [2, 4]],
        ...                                     index=['cat', 'dog'],
        ...                                     columns=multicol1)

        Stacking a dataframe with a multi-level column axis:

        >>> df_multi_level_cols1
             weight
                 kg    pounds
        cat       1        2
        dog       2        4
        >>> df_multi_level_cols1.stack()
                    weight
        cat kg           1
            pounds       2
        dog kg           2
            pounds       4

        **Missing values**

        >>> multicol2 = pd.MultiIndex.from_tuples([('weight', 'kg'),
        ...                                        ('height', 'm')])
        >>> df_multi_level_cols2 = cudf.DataFrame([[1.0, 2.0], [3.0, 4.0]],
        ...                                     index=['cat', 'dog'],
        ...                                     columns=multicol2)

        It is common to have missing values when stacking a dataframe
        with multi-level columns, as the stacked dataframe typically
        has more values than the original dataframe. Missing values
        are filled with NULLs:

        >>> df_multi_level_cols2
            weight height
                kg      m
        cat    1.0    2.0
        dog    3.0    4.0
        >>> df_multi_level_cols2.stack()
               weight height
        cat kg    1.0   <NA>
            m    <NA>    2.0
        dog kg    3.0   <NA>
            m    <NA>    4.0

        **Prescribing the level(s) to be stacked**

        The first parameter controls which level or levels are stacked:

        >>> df_multi_level_cols2.stack(0)
                    kg     m
        cat height  <NA>   2.0
            weight   1.0  <NA>
        dog height  <NA>   4.0
            weight   3.0  <NA>

        >>> df_multi_level_cols2.stack([0, 1])
        cat  height  m     2.0
             weight  kg    1.0
        dog  height  m     4.0
             weight  kg    3.0
        dtype: float64
        """
        if future_stack:
            if dropna is not no_default:
                raise ValueError(
                    "dropna must be unspecified with future_stack=True as "
                    "the new implementation does not introduce rows of NA "
                    "values. This argument will be removed in a future "
                    "version of cudf."
                )
        else:
            if dropna is not no_default or self._data.nlevels > 1:
                warnings.warn(
                    "The previous implementation of stack is deprecated and "
                    "will be removed in a future version of cudf. Specify "
                    "future_stack=True to adopt the new implementation and "
                    "silence this warning.",
                    FutureWarning,
                )
            if dropna is no_default:
                dropna = True

        if isinstance(level, (int, str)):
            level = [level]
        elif isinstance(level, list):
            if not all(isinstance(lv, (int, str)) for lv in level):
                raise ValueError(
                    "level must be either an int/str, or a list of int/str."
                )
        else:
            raise ValueError(
                "level must be either an int/str, or a list of int/str."
            )

        level = [level] if not isinstance(level, list) else level

        if not future_stack and len(level) > 1 and not dropna:
            raise NotImplementedError(
                "When stacking multiple levels, setting `dropna` to False "
                "will generate new column combination that does not exist "
                "in original dataframe. This behavior is unsupported in "
                "cuDF. See pandas deprecation note: "
                "https://github.com/pandas-dev/pandas/issues/53515"
            )

        # Compute the columns to stack based on specified levels

        level_indices: list[int] = []

        # If all passed in level names match up to the dataframe column's level
        # names, cast them to indices
        if all(lv in self._data.level_names for lv in level):
            level_indices = [self._data.level_names.index(lv) for lv in level]
        elif not all(isinstance(lv, int) for lv in level):
            raise ValueError(
                "`level` must either be a list of names or positions, not a "
                "mixture of both."
            )
        else:
            # Must be a list of positions, normalize negative positions
            level_indices = [
                lv + self._data.nlevels if lv < 0 else lv for lv in level
            ]

        unnamed_levels_indices = [
            i for i in range(self._data.nlevels) if i not in level_indices
        ]
        has_unnamed_levels = len(unnamed_levels_indices) > 0

        column_name_idx = self._data.to_pandas_index()
        # Construct new index from the levels specified by `level`
        named_levels = pd.MultiIndex.from_arrays(
            [column_name_idx.get_level_values(lv) for lv in level_indices]
        )

        # Since `level` may only specify a subset of all levels, `unique()` is
        # required to remove duplicates. In pandas, the order of the keys in
        # the specified levels are always sorted.
        unique_named_levels = named_levels.unique()
        if not future_stack:
            unique_named_levels = unique_named_levels.sort_values()

        # Each index from the original dataframe should repeat by the number
        # of unique values in the named_levels
        repeated_index = self.index.repeat(len(unique_named_levels))

        # Each column name should tile itself by len(df) times
        tiled_index = libcudf.reshape.tile(
            [
                as_column(unique_named_levels.get_level_values(i))
                for i in range(unique_named_levels.nlevels)
            ],
            self.shape[0],
        )

        # Assemble the final index
        new_index_columns = [*repeated_index._columns, *tiled_index]
        index_names = [*self._index.names, *unique_named_levels.names]
        new_index = MultiIndex.from_frame(
            DataFrame._from_data(
                dict(zip(range(0, len(new_index_columns)), new_index_columns))
            ),
            names=index_names,
        )

        # Compute the column indices that serves as the input for
        # `interleave_columns`
        column_idx_df = pd.DataFrame(
            data=range(len(self._data)), index=named_levels
        )

        column_indices: list[list[int]] = []
        if has_unnamed_levels:
            unnamed_level_values = list(
                map(column_name_idx.get_level_values, unnamed_levels_indices)
            )
            unnamed_level_values = pd.MultiIndex.from_arrays(
                unnamed_level_values
            )

        def unnamed_group_generator():
            if has_unnamed_levels:
                for _, grpdf in column_idx_df.groupby(by=unnamed_level_values):
                    # When stacking part of the levels, some combinations
                    # of keys may not be present in this group but can be
                    # present in others. Reindexing with the globally computed
                    # `unique_named_levels` assigns -1 to these key
                    # combinations, representing an all-null column that
                    # is used in the subsequent libcudf call.
                    if future_stack:
                        yield grpdf.reindex(
                            unique_named_levels, axis=0, fill_value=-1
                        ).values
                    else:
                        yield (
                            grpdf.reindex(
                                unique_named_levels, axis=0, fill_value=-1
                            )
                            .sort_index()
                            .values
                        )
            else:
                if future_stack:
                    yield column_idx_df.values
                else:
                    yield column_idx_df.sort_index().values

        column_indices = list(unnamed_group_generator())

        # For each of the group constructed from the unnamed levels,
        # invoke `interleave_columns` to stack the values.
        stacked = []

        for column_idx in column_indices:
            # Collect columns based on indices, append None for -1 indices.
            columns = [
                None if i == -1 else self._data.select_by_index(i).columns[0]
                for i in column_idx
            ]

            # Collect datatypes and cast columns as that type
            common_type = np.result_type(
                *(col.dtype for col in columns if col is not None)
            )

            all_nulls = functools.cache(
                functools.partial(
                    column_empty, self.shape[0], common_type, masked=True
                )
            )

            # homogenize the dtypes of the columns
            homogenized = [
                col.astype(common_type) if col is not None else all_nulls()
                for col in columns
            ]

            stacked.append(libcudf.reshape.interleave_columns(homogenized))

        # Construct the resulting dataframe / series
        if not has_unnamed_levels:
            result = Series._from_data(
                data={None: stacked[0]}, index=new_index
            )
        else:
            if unnamed_level_values.nlevels == 1:
                unnamed_level_values = unnamed_level_values.get_level_values(0)
            unnamed_level_values = unnamed_level_values.unique()

            data = ColumnAccessor(
                dict(
                    zip(
                        unnamed_level_values,
                        [
                            stacked[i]
                            for i in unnamed_level_values.argsort().argsort()
                        ]
                        if not future_stack
                        else [
                            stacked[i] for i in unnamed_level_values.argsort()
                        ],
                    )
                ),
                isinstance(unnamed_level_values, pd.MultiIndex),
                unnamed_level_values.names,
            )

            result = DataFrame._from_data(data, index=new_index)

        if not future_stack and dropna:
            return result.dropna(how="all")
        else:
            return result

    @_cudf_nvtx_annotate
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
        cols = self._data.to_pandas_index()
        df = DataFrame(cupy.asfortranarray(cov)).set_index(cols)
        df._set_columns_like(self._data)
        return df

    def corr(self, method="pearson", min_periods=None):
        """Compute the correlation matrix of a DataFrame.

        Parameters
        ----------
        method : {'pearson', 'spearman'}, default 'pearson'
            Method used to compute correlation:

            - pearson : Standard correlation coefficient
            - spearman : Spearman rank correlation

        min_periods : int, optional
            Minimum number of observations required per pair of columns to
            have a valid result.

        Returns
        -------
        DataFrame
            The requested correlation matrix.
        """
        if method == "pearson":
            values = self.values
        elif method == "spearman":
            values = self.rank().values
        else:
            raise ValueError("method must be either 'pearson', 'spearman'")

        if min_periods is not None:
            raise NotImplementedError("Unsupported argument 'min_periods'")

        corr = cupy.corrcoef(values, rowvar=False)
        cols = self._data.to_pandas_index()
        df = DataFrame(cupy.asfortranarray(corr)).set_index(cols)
        df._set_columns_like(self._data)
        return df

    @_cudf_nvtx_annotate
    def to_struct(self, name=None):
        """
        Return a struct Series composed of the columns of the DataFrame.

        Parameters
        ----------
        name: optional
            Name of the resulting Series

        Notes
        -----
        Note: a copy of the columns is made.
        """
        if not all(isinstance(name, str) for name in self._data.names):
            warnings.warn(
                "DataFrame contains non-string column name(s). Struct column "
                "requires field name to be string. Non-string column names "
                "will be casted to string as the field name."
            )
        fields = {str(name): col.dtype for name, col in self._data.items()}
        col = StructColumn(
            data=None,
            dtype=cudf.StructDtype(fields=fields),
            children=tuple(col.copy(deep=True) for col in self._data.columns),
            size=len(self),
            offset=0,
        )
        return cudf.Series._from_data(
            cudf.core.column_accessor.ColumnAccessor({name: col}),
            index=self.index,
            name=name,
        )

    @_cudf_nvtx_annotate
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
        Index([0, 1, 2, 3], dtype='int64')
        """
        return self._data.to_pandas_index()

    def itertuples(self, index=True, name="Pandas"):
        """
        Iteration is unsupported.

        See :ref:`iteration <pandas-comparison/iteration>` for more
        information.
        """
        raise TypeError(
            "cuDF does not support iteration of DataFrame "
            "via itertuples. Consider using "
            "`.to_pandas().itertuples()` "
            "if you wish to iterate over namedtuples."
        )

    def iterrows(self):
        """
        Iteration is unsupported.

        See :ref:`iteration <pandas-comparison/iteration>` for more
        information.
        """
        raise TypeError(
            "cuDF does not support iteration of DataFrame "
            "via iterrows. Consider using "
            "`.to_pandas().iterrows()` "
            "if you wish to iterate over each row."
        )

    @_cudf_nvtx_annotate
    @copy_docstring(reshape.pivot)
    def pivot(self, *, columns, index=no_default, values=no_default):
        return cudf.core.reshape.pivot(
            self, index=index, columns=columns, values=values
        )

    @_cudf_nvtx_annotate
    @copy_docstring(reshape.pivot_table)
    def pivot_table(
        self,
        values=None,
        index=None,
        columns=None,
        aggfunc="mean",
        fill_value=None,
        margins=False,
        dropna=None,
        margins_name="All",
        observed=False,
        sort=True,
    ):
        return cudf.core.reshape.pivot_table(
            self,
            values=values,
            index=index,
            columns=columns,
            aggfunc=aggfunc,
            fill_value=fill_value,
            margins=margins,
            dropna=dropna,
            margins_name=margins_name,
            observed=observed,
            sort=sort,
        )

    @_cudf_nvtx_annotate
    @copy_docstring(reshape.unstack)
    def unstack(self, level=-1, fill_value=None):
        return cudf.core.reshape.unstack(
            self, level=level, fill_value=fill_value
        )

    @_cudf_nvtx_annotate
    def explode(self, column, ignore_index=False):
        """
        Transform each element of a list-like to a row, replicating index
        values.

        Parameters
        ----------
        column : str
            Column to explode.
        ignore_index : bool, default False
            If True, the resulting index will be labeled 0, 1, …, n - 1.

        Returns
        -------
        DataFrame

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({
        ...     "a": [[1, 2, 3], [], None, [4, 5]],
        ...     "b": [11, 22, 33, 44],
        ... })
        >>> df
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

        return super()._explode(column, ignore_index)

    def pct_change(
        self, periods=1, fill_method=no_default, limit=no_default, freq=None
    ):
        """
        Calculates the percent change between sequential elements
        in the DataFrame.

        Parameters
        ----------
        periods : int, default 1
            Periods to shift for forming percent change.
        fill_method : str, default 'ffill'
            How to handle NAs before computing percent changes.

            .. deprecated:: 24.04
                All options of `fill_method` are deprecated
                except `fill_method=None`.
        limit : int, optional
            The number of consecutive NAs to fill before stopping.
            Not yet implemented.

            .. deprecated:: 24.04
                `limit` is deprecated.
        freq : str, optional
            Increment to use from time series API.
            Not yet implemented.

        Returns
        -------
        DataFrame
        """
        if limit is not no_default:
            raise NotImplementedError("limit parameter not supported yet.")
        if freq is not None:
            raise NotImplementedError("freq parameter not supported yet.")
        elif fill_method not in {
            no_default,
            None,
            "ffill",
            "pad",
            "bfill",
            "backfill",
        }:
            raise ValueError(
                "fill_method must be one of None, 'ffill', 'pad', "
                "'bfill', or 'backfill'."
            )

        if fill_method not in (no_default, None) or limit is not no_default:
            # Do not remove until pandas 3.0 support is added.
            assert (
                PANDAS_LT_300
            ), "Need to drop after pandas-3.0 support is added."
            warnings.warn(
                "The 'fill_method' and 'limit' keywords in "
                f"{type(self).__name__}.pct_change are deprecated and will be "
                "removed in a future version. Either fill in any non-leading "
                "NA values prior to calling pct_change or specify "
                "'fill_method=None' to not fill NA values.",
                FutureWarning,
            )
        if fill_method is no_default:
            fill_method = "ffill"
        if limit is no_default:
            limit = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = self.fillna(method=fill_method, limit=limit)

        return data.diff(periods=periods) / data.shift(
            periods=periods, freq=freq
        )

    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ):
        return df_protocol.__dataframe__(
            self, nan_as_null=nan_as_null, allow_copy=allow_copy
        )

    def nunique(self, axis=0, dropna=True):
        """
        Count number of distinct elements in specified axis.
        Return Series with number of distinct elements. Can ignore NaN values.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            The axis to use. 0 or 'index' for row-wise, 1 or 'columns' for
            column-wise.
        dropna : bool, default True
            Don't include NaN in the counts.

        Returns
        -------
        Series

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'A': [4, 5, 6], 'B': [4, 1, 1]})
        >>> df.nunique()
        A    3
        B    2
        dtype: int64
        """
        if axis != 0:
            raise NotImplementedError("axis parameter is not supported yet.")

        return cudf.Series(super().nunique(dropna=dropna))

    def _sample_axis_1(
        self,
        n: int,
        weights: Optional[ColumnLike],
        replace: bool,
        random_state: np.random.RandomState,
        ignore_index: bool,
    ):
        if replace:
            # Since cuDF does not support multiple columns with same name,
            # sample with replace=True at axis 1 is unsupported.
            raise NotImplementedError(
                "Sample is not supported for axis 1/`columns` when"
                "`replace=True`."
            )

        sampled_column_labels = random_state.choice(
            self._column_names, size=n, replace=False, p=weights
        )

        result = self._get_columns_by_label(sampled_column_labels)
        if ignore_index:
            result.reset_index(drop=True)

        return result

    def _from_columns_like_self(
        self,
        columns: List[ColumnBase],
        column_names: abc.Iterable[str],
        index_names: Optional[List[str]] = None,
        *,
        override_dtypes: Optional[abc.Iterable[Optional[Dtype]]] = None,
    ) -> DataFrame:
        result = super()._from_columns_like_self(
            columns,
            column_names,
            index_names,
            override_dtypes=override_dtypes,
        )
        result._set_columns_like(self._data)
        return result

    @_cudf_nvtx_annotate
    def interleave_columns(self):
        """
        Interleave Series columns of a table into a single column.

        Converts the column major table `cols` into a row major column.

        Parameters
        ----------
        cols : input Table containing columns to interleave.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({0: ['A1', 'A2', 'A3'], 1: ['B1', 'B2', 'B3']})
        >>> df
            0   1
        0  A1  B1
        1  A2  B2
        2  A3  B3
        >>> df.interleave_columns()
        0    A1
        1    B1
        2    A2
        3    B2
        4    A3
        5    B3
        dtype: object

        Returns
        -------
        The interleaved columns as a single column

        .. pandas-compat::
            **DataFrame.interleave_columns**

            This method does not exist in pandas but it can be run
            as ``pd.Series(np.vstack(df.to_numpy()).reshape((-1,)))``.
        """
        if ("category" == self.dtypes).any():
            raise ValueError(
                "interleave_columns does not support 'category' dtype."
            )

        return self._constructor_sliced._from_data(
            {None: libcudf.reshape.interleave_columns([*self._columns])}
        )

    @_cudf_nvtx_annotate
    def eval(self, expr: str, inplace: bool = False, **kwargs):
        """Evaluate a string describing operations on DataFrame columns.

        Operates on columns only, not specific rows or elements.

        Parameters
        ----------
        expr : str
            The expression string to evaluate.
        inplace : bool, default False
            If the expression contains an assignment, whether to perform the
            operation inplace and mutate the existing DataFrame. Otherwise,
            a new DataFrame is returned.
        **kwargs
            Not supported.

        Returns
        -------
        DataFrame, Series, or None
            Series if a single column is returned (the typical use case),
            DataFrame if any assignment statements are included in
            ``expr``, or None if ``inplace=True``.


        Examples
        --------
        >>> df = cudf.DataFrame({'A': range(1, 6), 'B': range(10, 0, -2)})
        >>> df
           A   B
        0  1  10
        1  2   8
        2  3   6
        3  4   4
        4  5   2
        >>> df.eval('A + B')
        0    11
        1    10
        2     9
        3     8
        4     7
        dtype: int64

        Assignment is allowed though by default the original DataFrame is not
        modified.

        >>> df.eval('C = A + B')
           A   B   C
        0  1  10  11
        1  2   8  10
        2  3   6   9
        3  4   4   8
        4  5   2   7
        >>> df
           A   B
        0  1  10
        1  2   8
        2  3   6
        3  4   4
        4  5   2

        Use ``inplace=True`` to modify the original DataFrame.

        >>> df.eval('C = A + B', inplace=True)
        >>> df
           A   B   C
        0  1  10  11
        1  2   8  10
        2  3   6   9
        3  4   4   8
        4  5   2   7

        Multiple columns can be assigned to using multi-line expressions:

        >>> df.eval(
        ...     '''
        ... C = A + B
        ... D = A - B
        ... '''
        ... )
           A   B   C  D
        0  1  10  11 -9
        1  2   8  10 -6
        2  3   6   9 -3
        3  4   4   8  0
        4  5   2   7  3

        .. pandas-compat::
            **DataFrame.eval**

            * Additional kwargs are not supported.
            * Bitwise and logical operators are not dtype-dependent.
              Specifically, `&` must be used for bitwise operators on integers,
              not `and`, which is specifically for the logical and between
              booleans.
            * Only numerical types are currently supported.
            * Operators generally will not cast automatically. Users are
              responsible for casting columns to suitable types before
              evaluating a function.
            * Multiple assignments to the same name (i.e. a sequence of
              assignment statements where later statements are conditioned upon
              the output of earlier statements) is not supported.
        """
        if kwargs:
            raise ValueError(
                "Keyword arguments other than `inplace` are not supported"
            )

        # Have to use a regex match to avoid capturing ==, >=, or <=
        equals_sign_regex = "[^=><]=[^=]"
        includes_assignment = re.search(equals_sign_regex, expr) is not None

        # Check if there were multiple statements. Filter out empty lines.
        statements = tuple(filter(None, expr.strip().split("\n")))
        if len(statements) > 1 and any(
            re.search(equals_sign_regex, st) is None for st in statements
        ):
            raise ValueError(
                "Multi-line expressions are only valid if all expressions "
                "contain an assignment."
            )

        if not includes_assignment:
            if inplace:
                raise ValueError(
                    "Cannot operate inplace if there is no assignment"
                )
            return Series._from_data(
                {
                    None: libcudf.transform.compute_column(
                        [*self._columns], self._column_names, statements[0]
                    )
                }
            )

        targets = []
        exprs = []
        for st in statements:
            try:
                t, e = re.split("[^=]=[^=]", st)
            except ValueError as err:
                if "too many values" in str(err):
                    raise ValueError(
                        f"Statement {st} contains too many assignments ('=')"
                    )
                raise
            targets.append(t.strip())
            exprs.append(e.strip())

        cols = (
            libcudf.transform.compute_column(
                [*self._columns], self._column_names, e
            )
            for e in exprs
        )
        ret = self if inplace else self.copy(deep=False)
        for name, col in zip(targets, cols):
            ret._data[name] = col
        if not inplace:
            return ret

    def value_counts(
        self,
        subset=None,
        normalize=False,
        sort=True,
        ascending=False,
        dropna=True,
    ):
        """
        Return a Series containing counts of unique rows in the DataFrame.

        Parameters
        ----------
        subset: list-like, optional
            Columns to use when counting unique combinations.
        normalize: bool, default False
            Return proportions rather than frequencies.
        sort: bool, default True
            Sort by frequencies.
        ascending: bool, default False
            Sort in ascending order.
        dropna: bool, default True
            Don't include counts of rows that contain NA values.

        Returns
        -------
        Series

        Notes
        -----
        The returned Series will have a MultiIndex with one level per input
        column. By default, rows that contain any NA values are omitted from
        the result. By default, the resulting Series will be in descending
        order so that the first element is the most frequently-occurring row.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'num_legs': [2, 4, 4, 6],
        ...                    'num_wings': [2, 0, 0, 0]},
        ...                    index=['falcon', 'dog', 'cat', 'ant'])
        >>> df
                num_legs  num_wings
        falcon         2          2
        dog            4          0
        cat            4          0
        ant            6          0
        >>> df.value_counts().sort_index()
        num_legs  num_wings
        2         2            1
        4         0            2
        6         0            1
        Name: count, dtype: int64
        """
        if subset:
            diff = set(subset) - set(self._data)
            if len(diff) != 0:
                raise KeyError(f"columns {diff} do not exist")
        columns = list(self._data.names) if subset is None else subset
        result = (
            self.groupby(
                by=columns,
                dropna=dropna,
            )
            .size()
            .astype("int64")
        )
        if sort:
            result = result.sort_values(ascending=ascending)
        if normalize:
            result = result / result._column.sum()
        # Pandas always returns MultiIndex even if only one column.
        if not isinstance(result.index, MultiIndex):
            result.index = MultiIndex._from_data(result._index._data)
        result.name = "proportion" if normalize else "count"
        return result


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
    wrapped_func = getattr(IndexedFrame, op)

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
            output._data[name] = as_column(
                value, length=len(output), dtype="bool"
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


@_cudf_nvtx_annotate
def from_pandas(obj, nan_as_null=no_default):
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

    >>> psr = pd.Series(['a', 'b', 'c', 'd'], name='apple', dtype='str')
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
    Index([1, 2, 10, 20], dtype='int64')
    >>> gidx = cudf.from_pandas(pidx)
    >>> gidx
    Index([1, 2, 10, 20], dtype='int64')
    >>> type(gidx)
    <class 'cudf.core.index.Index'>
    >>> type(pidx)
    <class 'pandas.core.indexes.base.Index'>

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
    if nan_as_null is no_default:
        nan_as_null = (
            False if cudf.get_option("mode.pandas_compatible") else None
        )

    if isinstance(obj, pd.DataFrame):
        return DataFrame.from_pandas(obj, nan_as_null=nan_as_null)
    elif isinstance(obj, pd.Series):
        return Series.from_pandas(obj, nan_as_null=nan_as_null)
    # This carveout for cudf.pandas is undesirable, but fixes crucial issues
    # for core RAPIDS projects like cuML and cuGraph that rely on
    # `cudf.from_pandas`, so we allow it for now.
    elif (ret := getattr(obj, "_fsproxy_wrapped", None)) is not None:
        return ret
    elif isinstance(obj, pd.MultiIndex):
        return MultiIndex.from_pandas(obj, nan_as_null=nan_as_null)
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


@_cudf_nvtx_annotate
def merge(left, right, *args, **kwargs):
    if isinstance(left, Series):
        left = left.to_frame()
    return left.merge(right, *args, **kwargs)


# a bit of fanciness to inject docstring with left parameter
merge_doc = DataFrame.merge.__doc__
if merge_doc is not None:
    idx = merge_doc.find("right")
    merge.__doc__ = "".join(
        [
            merge_doc[:idx],
            "\n\tleft : Series or DataFrame\n\t",
            merge_doc[idx:],
        ]
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
        common = set(lhs._column_names) & set(rhs._column_names)
        common_x = {f"{x}_x": x for x in common}
        common_y = {f"{x}_y": x for x in common}
        for col in df._column_names:
            if col in common_x:
                lhs_out[common_x[col]] = df[col]
            elif col in common_y:
                rhs_out[common_y[col]] = df[col]
            elif col in lhs:
                lhs_out[col] = df[col]
            elif col in rhs:
                rhs_out[col] = df[col]

    return lhs_out, rhs_out


def _setitem_with_dataframe(
    input_df: DataFrame,
    replace_df: DataFrame,
    input_cols: Any = None,
    mask: Optional[ColumnBase] = None,
    ignore_index: bool = False,
):
    """
    This function sets item dataframes relevant columns with replacement df
    :param input_df: Dataframe to be modified inplace
    :param replace_df: Replacement DataFrame to replace values with
    :param input_cols: columns to replace in the input dataframe
    :param mask: boolean mask in case of masked replacing
    :param ignore_index: Whether to conduct index equality and reindex
    """

    if input_cols is None:
        input_cols = input_df._column_names

    if len(input_cols) != len(replace_df._column_names):
        raise ValueError(
            "Number of Input Columns must be same replacement Dataframe"
        )

    if (
        not ignore_index
        and len(input_df) != 0
        and not input_df.index.equals(replace_df.index)
    ):
        replace_df = replace_df.reindex(input_df.index)

    for col_1, col_2 in zip(input_cols, replace_df._column_names):
        if col_1 in input_df._column_names:
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
                input_df._insert(
                    loc=len(input_df._data),
                    name=col_1,
                    value=replace_df[col_2],
                )


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
        merged_index = cudf.core.index.Index._concat(indexes)
        return merged_index.drop_duplicates()


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
        names_list = range(len(series_list))

    return names_list


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
            if cols[idx].null_count != len(cols[idx]):
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
            categories[idx] = cudf.Series(
                concat_columns([col.categories for col in cols])
            )._column.unique()
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
                        ._set_categories(
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
                codes=build_column(
                    cols[name].base_data, dtype=cols[name].dtype
                ),
                mask=cols[name].base_mask,
                offset=cols[name].offset,
                size=cols[name].size,
            )


def _from_dict_create_index(indexlist, namelist, library):
    if len(namelist) > 1:
        index = library.MultiIndex.from_tuples(indexlist, names=namelist)
    else:
        index = library.Index(indexlist, name=namelist[0])
    return index
