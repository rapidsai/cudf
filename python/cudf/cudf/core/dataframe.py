# Copyright (c) 2018-2020, NVIDIA CORPORATION.
from __future__ import division, print_function

import inspect
import itertools
import numbers
import pickle
import sys
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from types import GeneratorType

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
from numba import cuda
from pandas._config import get_option
from pandas.api.types import is_dict_like
from pandas.io.formats import console
from pandas.io.formats.printing import pprint_thing

import cudf
from cudf import _lib as libcudf
from cudf._lib.null_mask import MaskState, create_null_mask
from cudf._lib.nvtx import annotate
from cudf.core import column, reshape
from cudf.core.abc import Serializable
from cudf.core.column import as_column, column_empty
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.frame import Frame
from cudf.core.groupby.groupby import DataFrameGroupBy
from cudf.core.index import Index, RangeIndex, as_index
from cudf.core.indexing import _DataFrameIlocIndexer, _DataFrameLocIndexer
from cudf.core.series import Series
from cudf.core.window import Rolling
from cudf.utils import applyutils, docutils, ioutils, queryutils, utils
from cudf.utils.docutils import copy_docstring
from cudf.utils.dtypes import (
    cudf_dtype_from_pydata_dtype,
    is_categorical_dtype,
    is_column_like,
    is_list_dtype,
    is_list_like,
    is_scalar,
    is_string_dtype,
    is_struct_dtype,
    numeric_normalize_types,
)
from cudf.utils.utils import OrderedColumnDict


def _unique_name(existing_names, suffix="_unique_name"):
    ret = suffix
    i = 1
    while ret in existing_names:
        ret = "%s_%d" % (suffix, i)
        i += 1
    return ret


def _reverse_op(fn):
    return {
        "add": "radd",
        "radd": "add",
        "sub": "rsub",
        "rsub": "sub",
        "mul": "rmul",
        "rmul": "mul",
        "mod": "rmod",
        "rmod": "mod",
        "pow": "rpow",
        "rpow": "pow",
        "floordiv": "rfloordiv",
        "rfloordiv": "floordiv",
        "truediv": "rtruediv",
        "rtruediv": "truediv",
        "__add__": "__radd__",
        "__radd__": "__add__",
        "__sub__": "__rsub__",
        "__rsub__": "__sub__",
        "__mul__": "__rmul__",
        "__rmul__": "__mul__",
        "__mod__": "__rmod__",
        "__rmod__": "__mod__",
        "__pow__": "__rpow__",
        "__rpow__": "__pow__",
        "__floordiv__": "__rfloordiv__",
        "__rfloordiv__": "__floordiv__",
        "__truediv__": "__rtruediv__",
        "__rtruediv__": "__truediv__",
    }[fn]


_cupy_nan_methods_map = {
    "min": "nanmin",
    "max": "nanmax",
    "sum": "nansum",
    "prod": "nanprod",
    "mean": "nanmean",
    "std": "nanstd",
    "var": "nanvar",
}


class DataFrame(Frame, Serializable):

    _internal_names = {"_data", "_index"}

    @annotate("DATAFRAME_INIT", color="blue", domain="cudf_python")
    def __init__(self, data=None, index=None, columns=None, dtype=None):
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

        Examples
        --------

        Build dataframe with ``__setitem__``:

        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df['key'] = [0, 1, 2, 3, 4]
        >>> df['val'] = [float(i + 10) for i in range(5)]  # insert column
        >>> print(df)
        key   val
        0    0  10.0
        1    1  11.0
        2    2  12.0
        3    3  13.0
        4    4  14.0

        Build DataFrame via dict of columns:

        >>> import cudf
        >>> import numpy as np
        >>> from datetime import datetime, timedelta

        >>> t0 = datetime.strptime('2018-10-07 12:00:00', '%Y-%m-%d %H:%M:%S')
        >>> n = 5
        >>> df = cudf.DataFrame({
        ... 'id': np.arange(n),
        ... 'datetimes': np.array(
        ... [(t0+ timedelta(seconds=x)) for x in range(n)])
        ... })
        >>> df
            id                datetimes
        0    0  2018-10-07T12:00:00.000
        1    1  2018-10-07T12:00:01.000
        2    2  2018-10-07T12:00:02.000
        3    3  2018-10-07T12:00:03.000
        4    4  2018-10-07T12:00:04.000

        Build DataFrame via list of rows as tuples:

        >>> import cudf
        >>> df = cudf.DataFrame([
        ... (5, "cats", "jump", np.nan),
        ... (2, "dogs", "dig", 7.5),
        ... (3, "cows", "moo", -2.1, "occasionally"),
        ... ])
        >>> df
        0     1     2     3             4
        0  5  cats  jump  null          None
        1  2  dogs   dig   7.5          None
        2  3  cows   moo  -2.1  occasionally

        Convert from a Pandas DataFrame:

        >>> import pandas as pd
        >>> import cudf
        >>> pdf = pd.DataFrame({'a': [0, 1, 2, 3],'b': [0.1, 0.2, None, 0.3]})
        >>> df = cudf.from_pandas(pdf)
        >>> df
        a b
        0 0 0.1
        1 1 0.2
        2 2 nan
        3 3 0.3
        """
        super().__init__()

        if isinstance(data, ColumnAccessor):
            self._data = data
            if index is None:
                index = as_index(range(self._data.nrows))
            self._index = as_index(index)
            return None

        if isinstance(data, DataFrame):
            self._data = data._data
            self._index = data._index
            self.columns = data.columns
            return

        if isinstance(data, pd.DataFrame):
            data = self.from_pandas(data)
            self._data = data._data
            self._index = data._index
            self.columns = data.columns
            return

        if data is None:
            if index is None:
                self._index = RangeIndex(0)
            else:
                self._index = as_index(index)
            if columns is not None:
                if isinstance(columns, (Series, cudf.Index)):
                    columns = columns.to_pandas()

                self._data = ColumnAccessor(
                    OrderedDict.fromkeys(
                        columns,
                        column.column_empty(
                            len(self), dtype="object", masked=True
                        ),
                    )
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
            self.columns = new_df.columns
        elif hasattr(data, "__array_interface__"):
            arr_interface = data.__array_interface__
            if len(arr_interface["descr"]) == 1:
                # not record arrays
                new_df = self._from_arrays(data, index=index, columns=columns)
            else:
                new_df = self.from_records(data, index=index, columns=columns)
            self._data = new_df._data
            self._index = new_df._index
            self.columns = new_df.columns
        else:
            if is_list_like(data):
                if len(data) > 0 and is_scalar(data[0]):
                    new_df = self._from_columns(
                        [data], index=index, columns=columns
                    )
                    self._data = new_df._data
                    self._index = new_df._index
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

                self._init_from_dict_like(data, index=index, columns=columns)

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
            self._index = final_columns

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
        if columns:
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
        data = list(itertools.zip_longest(*data))

        if columns is not None and len(data) == 0:
            data = [
                cudf.core.column.column_empty(row_count=0, dtype=None)
                for _ in columns
            ]

        for col_name, col in enumerate(data):
            self._data[col_name] = column.as_column(col)

        self.columns = columns

    def _init_from_dict_like(self, data, index=None, columns=None):
        data = data.copy()
        num_rows = 0

        if columns is not None:
            # remove all entries in `data` that are
            # not in `columns`
            keys = [key for key in data.keys() if key in columns]
            data = {key: data[key] for key in keys}
            extra_cols = [col for col in columns if col not in data.keys()]
            if keys:
                # if keys is non-empty,
                # add null columns for all values
                # in `columns` that don't exist in `keys`:
                data.update({key: None for key in extra_cols})
            else:
                # if keys is empty,
                # it means that none of the actual keys in `data`
                # matches with `columns`.
                # Hence only assign `data` with `columns` as keys
                # and their values as empty columns.
                data = {
                    key: cudf.core.column.column_empty(row_count=0, dtype=None)
                    for key in extra_cols
                }

        data, index = self._align_input_series_indices(data, index=index)

        if index is None:
            if data:
                col_name = next(iter(data))
                if is_scalar(data[col_name]):
                    num_rows = num_rows or 1
                else:
                    data[col_name] = column.as_column(data[col_name])
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
                self.insert(i, col_name, data[col_name])

        if columns is not None:
            self.columns = columns

    @classmethod
    def _from_table(cls, table, index=None):
        if index is None:
            if table._index is not None:
                index = Index._from_table(table._index)
            else:
                index = RangeIndex(table._num_rows)
        out = cls.__new__(cls)
        out._data = table._data
        out._index = index
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
        header = {}
        frames = []
        header["type-serialized"] = pickle.dumps(type(self))
        header["index"], index_frames = self._index.serialize()
        header["index_frame_count"] = len(index_frames)
        frames.extend(index_frames)

        # Use the column directly to avoid duplicating the index
        # need to pickle column names to handle numpy integer columns
        header["column_names"] = pickle.dumps(tuple(self._data.names))
        column_header, column_frames = column.serialize_columns(self._columns)
        header["columns"] = column_header
        frames.extend(column_frames)

        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        # Reconstruct the index
        index_frames = frames[: header["index_frame_count"]]

        idx_typ = pickle.loads(header["index"]["type-serialized"])
        index = idx_typ.deserialize(header["index"], index_frames)

        # Reconstruct the columns
        column_frames = frames[header["index_frame_count"] :]

        column_names = pickle.loads(header["column_names"])
        columns = column.deserialize_columns(header["columns"], column_frames)

        return cls(dict(zip(column_names, columns)), index=index)

    @property
    def dtypes(self):
        """Return the dtypes in this object."""
        return pd.Series(
            [x.dtype for x in self._data.columns], index=self._data.names
        )

    @property
    def shape(self):
        """Returns a tuple representing the dimensionality of the DataFrame.
        """
        return self._num_rows, self._num_columns

    @property
    def ndim(self):
        """Dimension of the data. DataFrame ndim is always 2.
        """
        return 2

    def __dir__(self):
        o = set(dir(type(self)))
        o.update(self.__dict__)
        o.update(
            c for c in self.columns if isinstance(c, str) and c.isidentifier()
        )
        return list(o)

    def __setattr__(self, key, col):

        # if an attribute already exists, set it.
        try:
            object.__getattribute__(self, key)
            object.__setattr__(self, key, col)
            return
        except AttributeError:
            pass

        # if a column already exists, set it.
        if key not in self._internal_names:
            try:
                self[key]  # __getitem__ to verify key exists
                self[key] = col
                return
            except KeyError:
                pass

        object.__setattr__(self, key, col)

    def __getattr__(self, key):
        if key in self._internal_names:
            return object.__getattribute__(self, key)
        else:
            if key in self:
                return self[key]

        raise AttributeError("'DataFrame' object has no attribute %r" % key)

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
        >>> print(df[:4])    # get first 4 rows of all columns
           a  b  c
        0  0  0  0
        1  1  1  1
        2  2  2  2
        3  3  3  3
        >>> print(df[-5:])  # get last 5 rows of all columns
            a   b   c
        15  15  15  15
        16  16  16  16
        17  17  17  17
        18  18  18  18
        19  19  19  19
        >>> print(df[['a', 'c']]) # get columns a and c
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
        >>> print(df[[True, False, True, False]]) # mask the entire dataframe,
        # returning the rows specified in the boolean mask
        """
        if is_scalar(arg) or isinstance(arg, tuple):
            return self._get_columns_by_label(arg, downcast=True)

        elif isinstance(arg, slice):
            return self._slice(arg)

        elif isinstance(
            arg,
            (
                list,
                cupy.ndarray,
                np.ndarray,
                pd.Series,
                Series,
                Index,
                pd.Index,
            ),
        ):
            mask = arg
            if isinstance(mask, list):
                mask = pd.Series(mask)
            if mask.dtype == "bool":
                return self._apply_boolean_mask(mask)
            else:
                return self._get_columns_by_label(mask)
        elif isinstance(arg, DataFrame):
            return self.where(arg)
        else:
            msg = "__getitem__ on type {!r} is not supported"
            raise TypeError(msg.format(type(arg)))

    @annotate("DATAFRAME_SETITEM", color="blue", domain="cudf_python")
    def __setitem__(self, arg, value):
        """Add/set column by *arg or DataFrame*
        """
        if isinstance(arg, DataFrame):
            # not handling set_item where arg = df & value = df
            if isinstance(value, DataFrame):
                msg = (
                    "__setitem__ with arg = {!r} and "
                    "value = {!r} is not supported"
                )
                raise TypeError(msg.format(type(value), type(arg)))
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

        elif isinstance(
            arg, (list, np.ndarray, pd.Series, Series, Index, pd.Index)
        ):
            mask = arg
            if isinstance(mask, list):
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
                        # we will raise a key error if col not in dataframe
                        # this behavior will make it
                        # consistent to pandas >0.21.0
                        if not is_scalar(value):
                            self._data[col] = column.as_column(value)
                        else:
                            self._data[col][:] = value

        else:
            msg = "__setitem__ on type {!r} is not supported"
            raise TypeError(msg.format(type(arg)))

    def __delitem__(self, name):
        """
        Drop the given column by *name*.
        """
        self._drop_column(name)

    def __sizeof__(self):
        columns = sum(col.__sizeof__() for col in self._data.columns)
        index = self._index.__sizeof__()
        return columns + index

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
        ind = list(self.columns)
        sizes = [col._memory_usage(deep=deep) for col in self._data.columns]
        if index:
            ind.append("Index")
            ind = cudf.Index(ind, dtype="str")
            sizes.append(self.index.memory_usage(deep=deep))
        return Series(sizes, index=ind)

    def __len__(self):
        """
        Returns the number of rows
        """
        return len(self.index)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        import cudf

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
            else:
                return cudf_func(*args, **kwargs)
        else:
            return NotImplemented

    @property
    def values(self):
        """
        Return a CuPy representation of the DataFrame.

        Only the values in the DataFrame will be returned, the axes labels will
        be removed.

        Returns
        -------
        out: cupy.ndarray
            The values of the DataFrame.
        """
        return cupy.asarray(self.as_gpu_matrix())

    def __array__(self, dtype=None):
        raise TypeError(
            "Implicit conversion to a host NumPy array via __array__ is not "
            "allowed, To explicitly construct a GPU matrix, consider using "
            ".as_gpu_matrix()\nTo explicitly construct a host "
            "matrix, consider using .as_matrix()"
        )

    def __arrow_array__(self, type=None):
        raise TypeError(
            "Implicit conversion to a host PyArrow Table via __arrow_array__ "
            "is not allowed, To explicitly construct a PyArrow Table, "
            "consider using .to_arrow()"
        )

    def _get_numeric_data(self):
        """ Return a dataframe with only numeric data types """
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
        >>> print(df)
           a  b
        0  0  3
        1  1  4
        2  2  5
        """
        new = self.copy()
        for k, v in kwargs.items():
            new[k] = v
        return new

    def head(self, n=5):
        """
        Returns the first n rows as a new DataFrame

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df['key'] = [0, 1, 2, 3, 4]
        >>> df['val'] = [float(i + 10) for i in range(5)]  # insert column
        >>> print(df.head(2))
           key   val
        0    0  10.0
        1    1  11.0
        """
        return self.iloc[:n]

    def tail(self, n=5):
        """
        Returns the last n rows as a new DataFrame

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df['key'] = [0, 1, 2, 3, 4]
        >>> df['val'] = [float(i + 10) for i in range(5)]  # insert column
        >>> print(df.tail(2))
           key   val
        3    3  13.0
        4    4  14.0
        """
        if n == 0:
            return self.iloc[0:0]

        return self.iloc[-n:]

    def to_string(self):
        """
        Convert to string

        cuDF uses Pandas internals for efficient string formatting.
        Set formatting options using pandas string formatting options and
        cuDF objects will print identically to Pandas objects.

        cuDF supports `null/None` as a value in any column type, which
        is transparently supported during this output process.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df['key'] = [0, 1, 2]
        >>> df['val'] = [float(i + 10) for i in range(3)]
        >>> df.to_string()
        '   key   val\\n0    0  10.0\\n1    1  11.0\\n2    2  12.0'
        """
        return self.__repr__()

    def __str__(self):
        return self.to_string()

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
                    dtype=dtype, errors=errors, copy=copy, **kwargs
                )

        return result

    def _repr_pandas025_formatting(self, ncols, nrows, dtype=None):
        """
        With Pandas > 0.25 there are some new conditional formatting for some
        datatypes and column/row configurations. This fixes most of them in
        context to match the expected Pandas repr of the same content.

        Examples
        --------
        >>> gdf.__repr__()
            0   ...  19
        0   46  ...  48
        ..  ..  ...  ..
        19  40  ...  29

        [20 rows x 20 columns]

        >>> nrows, ncols = _repr_pandas025_formatting(2, 2, dtype="category")
        >>> pd.options.display.max_rows = nrows
        >>> pd.options.display.max_columns = ncols
        >>> gdf.__repr__()
             0  ...  19
        0   46  ...  48
        ..  ..  ...  ..
        19  40  ...  29

        [20 rows x 20 columns]
        """
        ncols = 1 if ncols in [0, 2] and dtype == "datetime64[ns]" else ncols
        ncols = (
            1
            if ncols == 0
            and nrows == 1
            and dtype in ["int8", "str", "category"]
            else ncols
        )
        ncols = (
            1
            if nrows == 1
            and dtype in ["int8", "int16", "int64", "str", "category"]
            else ncols
        )
        ncols = 0 if ncols == 2 else ncols
        ncols = 19 if ncols in [20, 21] else ncols
        return ncols, nrows

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
            elif df._data[col].has_nulls:
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
                    if isinstance(self.index, cudf.MultiIndex)
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

    # unary, binary, rbinary, orderedcompare, unorderedcompare
    def _apply_op(self, fn, other=None, fill_value=None):

        result = DataFrame(index=self.index)

        def op(lhs, rhs):
            if fill_value is None:
                return getattr(lhs, fn)(rhs)
            else:
                return getattr(lhs, fn)(rhs, fill_value)

        if other is None:
            for col in self._data:
                result[col] = getattr(self[col], fn)()
            return result
        elif isinstance(other, Sequence):
            for k, col in enumerate(self._data):
                result[col] = getattr(self[col], fn)(other[k])
        elif isinstance(other, DataFrame):
            if fn in ("__eq__", "__ne__"):
                if not self.index.equals(other.index):
                    raise ValueError(
                        "Can only compare identically-labeled "
                        "DataFrame objects"
                    )

            lhs, rhs = _align_indices(self, other)
            result.index = lhs.index
            max_num_rows = max(lhs.shape[0], rhs.shape[0])

            def fallback(col, fn):
                if fill_value is None:
                    return Series.from_masked_array(
                        data=column_empty(max_num_rows, dtype="float64"),
                        mask=create_null_mask(
                            max_num_rows, state=MaskState.ALL_NULL
                        ),
                    ).set_index(col.index)
                else:
                    return getattr(col, fn)(fill_value)

            for col in lhs._data:
                if col not in rhs._data:
                    result[col] = fallback(lhs[col], fn)
                else:
                    result[col] = op(lhs[col], rhs[col])
            for col in rhs._data:
                if col not in lhs._data:
                    result[col] = fallback(rhs[col], _reverse_op(fn))
        elif isinstance(other, Series):
            other_cols = other.to_pandas().to_dict()
            other_cols_keys = list(other_cols.keys())
            result_cols = list(self.columns)
            df_cols = list(result_cols)
            for new_col in other_cols.keys():
                if new_col not in result_cols:
                    result_cols.append(new_col)
            for col in result_cols:
                if col in df_cols and col in other_cols_keys:
                    l_opr = self[col]
                    r_opr = other_cols[col]
                else:
                    if col not in df_cols:
                        r_opr = other_cols[col]
                        l_opr = Series(
                            column_empty(
                                len(self), masked=True, dtype=other.dtype
                            )
                        )
                    if col not in other_cols_keys:
                        r_opr = None
                        l_opr = self[col]
                result[col] = op(l_opr, r_opr)

        elif isinstance(other, numbers.Number):
            for col in self._data:
                result[col] = op(self[col], other)
        else:
            raise NotImplementedError(
                "DataFrame operations with " + str(type(other)) + " not "
                "supported at this time."
            )
        return result

    def add(self, other, axis="columns", level=None, fill_value=None):
        """
        Get Addition of dataframe and other, element-wise (binary
        operator `add`).

        Equivalent to ``dataframe + other``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `radd`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df + 1
                angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361
        >>> df.add(1)
                angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361
        """
        if axis not in (1, "columns"):
            raise NotImplementedError("Only axis=1 supported at this time.")

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._apply_op("add", other, fill_value)

    def __add__(self, other):
        return self._apply_op("__add__", other)

    def radd(self, other, axis=1, level=None, fill_value=None):
        """
        Get Addition of dataframe and other, element-wise (binary
        operator `radd`).

        Equivalent to ``other + dataframe``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `add`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df + 1
                angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361
        >>> df.radd(1)
                angles  degrees
        circle          1      361
        triangle        4      181
        rectangle       5      361
        """
        if axis not in (1, "columns"):
            raise NotImplementedError("Only axis=1 supported at this time.")

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._apply_op("radd", other, fill_value)

    def __radd__(self, other):
        return self._apply_op("__radd__", other)

    def sub(self, other, axis="columns", level=None, fill_value=None):
        """
        Get Subtraction of dataframe and other, element-wise (binary
        operator `sub`).

        Equivalent to ``dataframe - other``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `rsub`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df.sub(1)
                angles  degrees
        circle         -1      359
        triangle        2      179
        rectangle       3      359
        >>> df.sub([1, 2])
                angles  degrees
        circle         -1      358
        triangle        2      178
        rectangle       3      358
        """
        if axis not in (1, "columns"):
            raise NotImplementedError("Only axis=1 supported at this time.")

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._apply_op("sub", other, fill_value)

    def __sub__(self, other):
        return self._apply_op("__sub__", other)

    def rsub(self, other, axis="columns", level=None, fill_value=None):
        """
        Get Subtraction of dataframe and other, element-wise (binary
        operator `rsub`).

        Equivalent to ``other - dataframe``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `sub`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360
        >>> df.rsub(1)
                   angles  degrees
        circle          1     -359
        triangle       -2     -179
        rectangle      -3     -359
        >>> df.rsub([1, 2])
                   angles  degrees
        circle          1     -358
        triangle       -2     -178
        rectangle      -3     -358
        """
        if axis not in (1, "columns"):
            raise NotImplementedError("Only axis=1 supported at this time.")

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._apply_op("rsub", other, fill_value)

    def __rsub__(self, other):
        return self._apply_op("__rsub__", other)

    def mul(self, other, axis="columns", level=None, fill_value=None):
        """
        Get Multiplication of dataframe and other, element-wise (binary
        operator `mul`).

        Equivalent to ``dataframe * other``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `rmul`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> df * other
                angles degrees
        circle          0    null
        triangle        9    null
        rectangle      16    null
        >>> df.mul(other, fill_value=0)
                angles  degrees
        circle          0        0
        triangle        9        0
        rectangle      16        0
        """
        if axis not in (1, "columns"):
            raise NotImplementedError("Only axis=1 supported at this time.")

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._apply_op("mul", other, fill_value)

    def __mul__(self, other):
        return self._apply_op("__mul__", other)

    def rmul(self, other, axis="columns", level=None, fill_value=None):
        """
        Get Multiplication of dataframe and other, element-wise (binary
        operator `rmul`).

        Equivalent to ``other * dataframe``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `mul`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> other = pd.DataFrame({'angles': [0, 3, 4]},
        ...                      index=['circle', 'triangle', 'rectangle'])
        >>> other * df
                angles degrees
        circle          0    null
        triangle        9    null
        rectangle      16    null
        >>> df.rmul(other, fill_value=0)
                angles  degrees
        circle          0        0
        triangle        9        0
        rectangle      16        0
        """
        if axis not in (1, "columns"):
            raise NotImplementedError("Only axis=1 supported at this time.")

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._apply_op("rmul", other, fill_value)

    def __rmul__(self, other):
        return self._apply_op("__rmul__", other)

    def mod(self, other, axis="columns", level=None, fill_value=None):
        """
        Get Modulo division of dataframe and other, element-wise (binary
        operator `mod`).

        Equivalent to ``dataframe % other``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `rmod`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df % 100
                angles  degrees
        circle          0       60
        triangle        3       80
        rectangle       4       60
        >>> df.mod(100)
                angles  degrees
        circle          0       60
        triangle        3       80
        rectangle       4       60
        """
        if axis not in (1, "columns"):
            raise NotImplementedError("Only axis=1 supported at this time.")

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._apply_op("mod", other, fill_value)

    def __mod__(self, other):
        return self._apply_op("__mod__", other)

    def rmod(self, other, axis="columns", level=None, fill_value=None):
        """
        Get Modulo division of dataframe and other, element-wise (binary
        operator `rmod`).

        Equivalent to ``other % dataframe``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `mod`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [1, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> 100 % df
                angles  degrees
        circle          0      100
        triangle        1      100
        rectangle       0      100
        >>> df.rmod(100)
                angles  degrees
        circle          0      100
        triangle        1      100
        rectangle       0      100
        """
        if axis not in (1, "columns"):
            raise NotImplementedError("Only axis=1 supported at this time.")

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._apply_op("rmod", other, fill_value)

    def __rmod__(self, other):
        return self._apply_op("__rmod__", other)

    def pow(self, other, axis="columns", level=None, fill_value=None):
        """
        Get Exponential power of dataframe and other, element-wise (binary
        operator `pow`).

        Equivalent to ``dataframe ** other``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `rpow`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [1, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df ** 2
                angles  degrees
        circle          0   129600
        triangle        9    32400
        rectangle      16   129600
        >>> df.pow(2)
                angles  degrees
        circle          0   129600
        triangle        9    32400
        rectangle      16   129600
        """
        if axis not in (1, "columns"):
            raise NotImplementedError("Only axis=1 supported at this time.")

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._apply_op("pow", other, fill_value)

    def __pow__(self, other):
        return self._apply_op("__pow__", other)

    def rpow(self, other, axis="columns", level=None, fill_value=None):
        """
        Get Exponential power of dataframe and other, element-wise (binary
        operator `pow`).

        Equivalent to ``other ** dataframe``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `pow`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [1, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> 1 ** df
                angles  degrees
        circle          1        1
        triangle        1        1
        rectangle       1        1
        >>> df.rpow(1)
                angles  degrees
        circle          1        1
        triangle        1        1
        rectangle       1        1
        """
        if axis not in (1, "columns"):
            raise NotImplementedError("Only axis=1 supported at this time.")

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._apply_op("rpow", other, fill_value)

    def __rpow__(self, other):
        return self._apply_op("__pow__", other)

    def floordiv(self, other, axis="columns", level=None, fill_value=None):
        """
        Get Integer division of dataframe and other, element-wise (binary
        operator `floordiv`).

        Equivalent to ``dataframe // other``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `rfloordiv`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [1, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df.floordiv(2)
                angles  degrees
        circle          0      180
        triangle        1       90
        rectangle       2      180
        >>> df // 2
                angles  degrees
        circle          0      180
        triangle        1       90
        rectangle       2      180
        """
        if axis not in (1, "columns"):
            raise NotImplementedError("Only axis=1 supported at this time.")

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._apply_op("floordiv", other, fill_value)

    def __floordiv__(self, other):
        return self._apply_op("__floordiv__", other)

    def rfloordiv(self, other, axis="columns", level=None, fill_value=None):
        """
        Get Integer division of dataframe and other, element-wise (binary
        operator `rfloordiv`).

        Equivalent to ``other // dataframe``, but with support to substitute
        a fill_value for missing data in one of the inputs. With reverse
        version, `floordiv`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'col1': [10, 11, 23],
        ... 'col2': [101, 122, 321]})
        >>> df
           col1  col2
        0    10   101
        1    11   122
        2    23   321
        >>> df.rfloordiv(df)
           col1  col2
        0     1     1
        1     1     1
        2     1     1
        >>> df.rfloordiv(200)
           col1  col2
        0    20     1
        1    18     1
        2     8     0
        >>> df.rfloordiv(100)
           col1  col2
        0    10     0
        1     9     0
        2     4     0
        """

        if axis not in (1, "columns"):
            raise NotImplementedError("Only axis=1 supported at this time.")

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._apply_op("rfloordiv", other, fill_value)

    def __rfloordiv__(self, other):
        return self._apply_op("__rfloordiv__", other)

    def truediv(self, other, axis="columns", level=None, fill_value=None):
        """
        Get Floating division of dataframe and other, element-wise (binary
        operator `truediv`).

        Equivalent to ``dataframe / other``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `rtruediv`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df.truediv(10)
                    angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0
        >>> df.div(10)
                    angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0
        >>> df / 10
                    angles  degrees
        circle        0.0     36.0
        triangle      0.3     18.0
        rectangle     0.4     36.0
        """
        if axis not in (1, "columns"):
            raise NotImplementedError("Only axis=1 supported at this time.")

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._apply_op("truediv", other, fill_value)

    # Alias for truediv
    div = truediv

    def __truediv__(self, other):
        return self._apply_op("__truediv__", other)

    def rtruediv(self, other, axis="columns", level=None, fill_value=None):
        """
        Get Floating division of dataframe and other, element-wise (binary
        operator `rtruediv`).

        Equivalent to ``other / dataframe``, but with support to substitute a
        fill_value for missing data in one of the inputs. With reverse
        version, `truediv`.

        Among flexible wrappers (`add`, `sub`, `mul`, `div`, `mod`, `pow`) to
        arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`.

        Parameters
        ----------

        other : scalar, sequence, Series, or DataFrame
            Any single or multiple element data structure, or list-like object.
        fill_value  : float or None, default None
            Fill existing missing (NaN) values, and any new element needed
            for successful DataFrame alignment, with this value before
            computation. If data in both corresponding DataFrame locations
            is missing the result will be missing.

        Returns
        -------
        DataFrame
            Result of the arithmetic operation.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'angles': [0, 3, 4],
        ...                    'degrees': [360, 180, 360]},
        ...                   index=['circle', 'triangle', 'rectangle'])
        >>> df
                   angles  degrees
        circle          0      360
        triangle        3      180
        rectangle       4      360
        >>> df.rtruediv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778
        >>> df.rdiv(10)
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778
        >>> 10 / df
                     angles   degrees
        circle          inf  0.027778
        triangle   3.333333  0.055556
        rectangle  2.500000  0.027778
        """
        if axis not in (1, "columns"):
            raise NotImplementedError("Only axis=1 supported at this time.")

        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        return self._apply_op("rtruediv", other, fill_value)

    # Alias for rtruediv
    rdiv = rtruediv

    def __rtruediv__(self, other):
        return self._apply_op("__rtruediv__", other)

    __div__ = __truediv__

    def __and__(self, other):
        return self._apply_op("__and__", other)

    def __or__(self, other):
        return self._apply_op("__or__", other)

    def __xor__(self, other):
        return self._apply_op("__xor__", other)

    def __eq__(self, other):
        return self._apply_op("__eq__", other)

    def __ne__(self, other):
        return self._apply_op("__ne__", other)

    def __lt__(self, other):
        return self._apply_op("__lt__", other)

    def __le__(self, other):
        return self._apply_op("__le__", other)

    def __gt__(self, other):
        return self._apply_op("__gt__", other)

    def __ge__(self, other):
        return self._apply_op("__ge__", other)

    def __invert__(self):
        return self._apply_op("__invert__")

    def __neg__(self):
        return self._apply_op("__neg__")

    def __abs__(self):
        return self._apply_op("__abs__")

    def __iter__(self):
        return iter(self.columns)

    def iteritems(self):
        """ Iterate over column names and series pairs """
        for k in self:
            yield (k, self[k])

    @property
    @annotate("DATAFRAME_LOC", color="blue", domain="cudf_python")
    def loc(self):
        """
        Selecting rows and columns by label or boolean mask.

        Examples
        --------

        DataFrame with string index.

        >>> print(df)
           a  b
        a  0  5
        b  1  6
        c  2  7
        d  3  8
        e  4  9

        Select a single row by label.

        >>> print(df.loc['a'])
        a    0
        b    5
        Name: a, dtype: int64

        Select multiple rows and a single column.

        >>> print(df.loc[['a', 'c', 'e'], 'b'])
        a    5
        c    7
        e    9
        Name: b, dtype: int64

        Selection by boolean mask.

        >>> print(df.loc[df.a > 2])
           a  b
        d  3  8
        e  4  9

        Setting values using loc.

        >>> df.loc[['a', 'c', 'e'], 'a'] = 0
        >>> print(df)
           a  b
        a  0  5
        b  1  6
        c  0  7
        d  3  8
        e  0  9

        See also
        --------
        DataFrame.iloc

        Notes
        -----
        One notable difference from Pandas is when DataFrame is of
        mixed types and result is expected to be a Series in case of Pandas.
        cuDF will return a DataFrame as it doesn't support mixed types
        under Series yet.

        Mixed dtype single row output as a dataframe (pandas results in Series)

        >>> import cudf
        >>> df = cudf.DataFrame({"a":[1, 2, 3], "b":["a", "b", "c"]})
        >>> df.loc[0]
           a  b
        0  1  a
        """
        return _DataFrameLocIndexer(self)

    @property
    def iloc(self):
        """
        Selecting rows and column by position.

        Examples
        --------
        >>> df = cudf.DataFrame({'a': range(20),
        ...                      'b': range(20),
        ...                      'c': range(20)})

        Select a single row using an integer index.

        >>> print(df.iloc[1])
        a    1
        b    1
        c    1

        Select multiple rows using a list of integers.

        >>> print(df.iloc[[0, 2, 9, 18]])
              a    b    c
         0    0    0    0
         2    2    2    2
         9    9    9    9
        18   18   18   18

        Select rows using a slice.

        >>> print(df.iloc[3:10:2])
             a    b    c
        3    3    3    3
        5    5    5    5
        7    7    7    7
        9    9    9    9

        Select both rows and columns.

        >>> print(df.iloc[[1, 3, 5, 7], 2])
        1    1
        3    3
        5    5
        7    7
        Name: c, dtype: int64

        Setting values in a column using iloc.

        >>> df.iloc[:4] = 0
        >>> print(df)
           a  b  c
        0  0  0  0
        1  0  0  0
        2  0  0  0
        3  0  0  0
        4  4  4  4
        5  5  5  5
        6  6  6  6
        7  7  7  7
        8  8  8  8
        9  9  9  9
        [10 more rows]

        See also
        --------
        DataFrame.loc

        Notes
        -----
        One notable difference from Pandas is when DataFrame is of
        mixed types and result is expected to be a Series in case of Pandas.
        cuDF will return a DataFrame as it doesn't support mixed types
        under Series yet.

        Mixed dtype single row output as a dataframe (pandas results in Series)

        >>> import cudf
        >>> df = cudf.DataFrame({"a":[1, 2, 3], "b":["a", "b", "c"]})
        >>> df.iloc[0]
           a  b
        0  1  a
        """
        return _DataFrameIlocIndexer(self)

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

    @property
    @annotate("DATAFRAME_COLUMNS_GETTER", color="yellow", domain="cudf_python")
    def columns(self):
        """Returns a tuple of columns
        """
        return self._data.to_pandas_index()

    @columns.setter
    @annotate("DATAFRAME_COLUMNS_SETTER", color="yellow", domain="cudf_python")
    def columns(self, columns):
        if isinstance(columns, (cudf.MultiIndex, cudf.Index)):
            columns = columns.to_pandas()
        if columns is None:
            columns = pd.Index(range(len(self._data.columns)))
        is_multiindex = isinstance(columns, pd.MultiIndex)

        if isinstance(
            columns, (Series, cudf.Index, cudf.core.column.ColumnBase)
        ):
            columns = pd.Index(columns.to_array(), tupleize_cols=is_multiindex)
        elif not isinstance(columns, pd.Index):
            columns = pd.Index(columns, tupleize_cols=is_multiindex)

        if not len(columns) == len(self._data.names):
            raise ValueError(
                f"Length mismatch: expected {len(self._data.names)} elements ,"
                f"got {len(columns)} elements"
            )

        data = dict(zip(columns, self._data.columns))
        if len(columns) != len(data):
            raise ValueError("Duplicate column names are not allowed")

        self._data = ColumnAccessor(
            data, multiindex=is_multiindex, level_names=columns.names,
        )

    def _rename_columns(self, new_names):
        old_cols = iter(self._data.names)
        l_old_cols = len(self._data)
        l_new_cols = len(new_names)
        if l_new_cols != l_old_cols:
            msg = (
                f"Length of new column names: {l_new_cols} does not "
                "match length of previous column names: {l_old_cols}"
            )
            raise ValueError(msg)

        mapper = dict(zip(old_cols, new_names))
        self.rename(mapper=mapper, inplace=True, axis=1)

    @property
    def index(self):
        """Returns the index of the DataFrame
        """
        return self._index

    @index.setter
    def index(self, value):
        if isinstance(value, cudf.core.multiindex.MultiIndex):
            if len(self._data) > 0 and len(value) != len(self):
                msg = (
                    f"Length mismatch: Expected axis has {len(self)} "
                    f"elements, new values have {len(value)} elements"
                )
                raise ValueError(msg)
            self._index = value
            return

        new_length = len(value)
        old_length = len(self._index)

        if len(self._data) > 0 and new_length != old_length:
            msg = (
                f"Length mismatch: Expected axis has {old_length} elements, "
                f"new values have {new_length} elements"
            )
            raise ValueError(msg)

        # try to build an index from generic _index
        idx = as_index(value)
        self._index = idx

    def reindex(
        self, labels=None, axis=0, index=None, columns=None, copy=True
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
        >>> print(df)
           key   val
        0    0  10.0
        1    1  11.0
        2    2  12.0
        3    3  13.0
        4    4  14.0
        >>> print(df_new)
           key   val  sum
        0    0  10.0  NaN
        3    3  13.0  NaN
        4    4  14.0  NaN
        5   -1   NaN  NaN
        """

        if labels is None and index is None and columns is None:
            return self.copy(deep=copy)

        df = self
        cols = columns
        original_cols = df._data
        dtypes = OrderedDict(df.dtypes)
        idx = labels if index is None and axis in (0, "index") else index
        cols = labels if cols is None and axis in (1, "columns") else cols
        df = df if cols is None else df[list(set(df.columns) & set(cols))]

        if idx is not None:
            idx = as_index(idx)

            if isinstance(idx, cudf.core.MultiIndex):
                idx_dtype_match = (
                    df.index._source_data.dtypes == idx._source_data.dtypes
                ).all()
            else:
                idx_dtype_match = df.index.dtype == idx.dtype

            if not idx_dtype_match:
                cols = cols if cols is not None else list(df.columns)
                df = DataFrame()
            else:
                df = DataFrame(None, idx).join(df, how="left", sort=True)
                # double-argsort to map back from sorted to unsorted positions
                df = df.take(idx.argsort(ascending=True).argsort())

        idx = idx if idx is not None else df.index
        names = cols if cols is not None else list(df.columns)

        length = len(idx)
        cols = OrderedDict()

        for name in names:
            if name in df:
                cols[name] = df._data[name].copy(deep=copy)
            else:
                dtype = dtypes.get(name, np.float64)
                col = original_cols.get(name, Series(dtype=dtype)._column)
                col = column.column_empty_like(
                    col, dtype=dtype, masked=True, newsize=length
                )
                cols[name] = col

        return DataFrame(cols, idx)

    def _set_index(
        self, index, to_drop=None, inplace=False, verify_integrity=False,
    ):
        """Helper for `.set_index`

        Parameters
        ----------
        index : Index
            The new index to set.
        to_drop : list optional, default None
            A list of labels indicating columns to drop.
        inplace : boolean, default False
            Modify the DataFrame in place (do not create a new object).
        verify_integrity : boolean, default False
            Check for duplicates in the new index.
        """
        if not isinstance(index, Index):
            raise ValueError("Parameter index should be type `Index`.")

        df = self if inplace else self.copy(deep=True)

        if verify_integrity and not index.is_unique:
            raise ValueError(f"Values in Index are not unique: {index}")

        if to_drop:
            df.drop(columns=to_drop, inplace=True)

        df.index = index
        return df if not inplace else None

    def set_index(
        self,
        index,
        drop=True,
        append=False,
        inplace=False,
        verify_integrity=False,
    ):
        """Return a new DataFrame with a new index

        Parameters
        ----------
        index : Index, Series-convertible, label-like, or list
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
        >>> df = cudf.DataFrame({"a": [1, 2, 3, 4, 5],
        ... "b": ["a", "b", "c", "d","e"],
        ... "c": [1.0, 2.0, 3.0, 4.0, 5.0]})
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

        if not isinstance(index, list):
            index = [index]

        # Preliminary type check
        col_not_found = []
        columns_to_add = []
        names = []
        to_drop = []
        for i, col in enumerate(index):
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
                if isinstance(col, (cudf.MultiIndex, pd.MultiIndex)):
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
            if isinstance(self.index, cudf.MultiIndex):
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
            idf = cudf.DataFrame()
            for i, col in enumerate(columns_to_add):
                idf[i] = col
            idx = cudf.MultiIndex.from_frame(idf, names=names)

        return self._set_index(
            index=idx,
            to_drop=to_drop,
            inplace=inplace,
            verify_integrity=verify_integrity,
        )

    def reset_index(
        self, level=None, drop=False, inplace=False, col_level=0, col_fill=""
    ):
        """
        Reset the index.

        Reset the index of the DataFrame, and use the default one instead.

        Parameters
        ----------
        drop : bool, default False
            Do not try to insert index into dataframe columns. This resets
            the index to the default integer index.
        inplace : bool, default False
            Modify the DataFrame in place (do not create a new object).

        Returns
        -------
        DataFrame or None
            DataFrame with the new index or None if ``inplace=True``.

        Examples
        --------
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
        monkey  mammal      null
        >>> df.reset_index()
            index   class max_speed
        0  falcon    bird     389.0
        1  parrot    bird      24.0
        2    lion  mammal      80.5
        3  monkey  mammal      null
        >>> df.reset_index(drop=True)
            class max_speed
        0    bird     389.0
        1    bird      24.0
        2  mammal      80.5
        3  mammal      null
        """
        if level is not None:
            raise NotImplementedError("level parameter is not supported yet.")

        if col_level != 0:
            raise NotImplementedError(
                "col_level parameter is not supported yet."
            )

        if col_fill != "":
            raise NotImplementedError(
                "col_fill parameter is not supported yet."
            )

        if inplace:
            result = self
        else:
            result = self.copy()
        if all(name is None for name in self.index.names):
            if isinstance(self.index, cudf.MultiIndex):
                names = tuple(
                    f"level_{i}" for i, _ in enumerate(self.index.names)
                )
            else:
                names = ("index",)
        else:
            names = self.index.names

        if not drop:
            index_columns = self.index._data.columns
            for name, index_column in zip(
                reversed(names), reversed(index_columns)
            ):
                result.insert(0, name, index_column)
        result.index = RangeIndex(len(self))
        if inplace:
            return
        else:
            return result

    def take(self, positions, keep_index=True):
        """
        Return a new DataFrame containing the rows specified by *positions*

        Parameters
        ----------
        positions : array-like
            Integer or boolean array-like specifying the rows of the output.
            If integer, each element represents the integer index of a row.
            If boolean, *positions* must be of the same length as *self*,
            and represents a boolean mask.

        Returns
        -------
        out : DataFrame
            New DataFrame

        Examples
        --------
        >>> a = cudf.DataFrame({'a': [1.0, 2.0, 3.0],
        ...                    'b': cudf.Series(['a', 'b', 'c'])})
        >>> a.take([0, 2, 2])
             a  b
        0  1.0  a
        2  3.0  c
        2  3.0  c
        >>> a.take([True, False, True])
             a  b
        0  1.0  a
        2  3.0  c
        """
        positions = as_column(positions)
        if pd.api.types.is_bool_dtype(positions):
            return self._apply_boolean_mask(positions)
        out = self._gather(positions, keep_index=keep_index)
        out.columns = self.columns
        return out

    @annotate("DATAFRAME_COPY", color="cyan", domain="cudf_python")
    def copy(self, deep=True):
        """
        Returns a copy of this dataframe

        Parameters
        ----------
        deep: bool
           Make a full copy of Series columns and Index at the GPU level, or
           create a new allocation with references.
        """
        out = DataFrame(data=self._data.copy(deep=deep))
        out.index = self.index.copy(deep=deep)
        return out

    def __copy__(self):
        return self.copy(deep=True)

    def __deepcopy__(self, memo=None):
        """
        Parameters
        ----------
        memo, default None
            Standard signature. Unused
        """
        if memo is None:
            memo = {}
        return self.copy(deep=True)

    @annotate("INSERT", color="green", domain="cudf_python")
    def insert(self, loc, name, value):
        """ Add a column to DataFrame at the index specified by loc.

        Parameters
        ----------
        loc : int
            location to insert by index, cannot be greater then num columns + 1
        name : number or string
            name or label of column to be inserted
        value : Series or array-like
        """
        num_cols = len(self._data)
        if name in self._data:
            raise NameError("duplicated column name {!r}".format(name))

        if loc < 0:
            loc = num_cols + loc + 1

        if not (0 <= loc <= num_cols):
            raise ValueError(
                "insert location must be within range {}, {}".format(
                    -(num_cols + 1) * (num_cols > 0), num_cols * (num_cols > 0)
                )
            )

        if is_scalar(value):
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
            value = Series(value)._align_to_index(
                self._index, how="right", sort=False
            )

        value = column.as_column(value)

        self._data.insert(name, value, loc=loc)

    def add_column(self, name, data, forceindex=False):
        """Add a column

        Parameters
        ----------
        name : str
            Name of column to be added.
        data : Series, array-like
            Values to be added.
        """

        warnings.warn(
            "`add_column` will be removed in the future. Use `.insert`",
            DeprecationWarning,
        )

        if name in self._data:
            raise NameError("duplicated column name {!r}".format(name))

        if isinstance(data, GeneratorType):
            data = Series(data)

        self.insert(len(self._data.names), name, data)

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
            outdf = self
        else:
            outdf = self.copy()

        if axis in (1, "columns"):
            target = _get_host_unique(target)

            _drop_columns(outdf, target, errors)
        elif axis in (0, "index"):
            if not isinstance(target, (cudf.Series, cudf.Index)):
                target = column.as_column(target)

            if isinstance(self._index, cudf.MultiIndex):
                if level is None:
                    level = 0

                levels_index = outdf.index.get_level_values(level)
                if errors == "raise" and not target.isin(levels_index).all():
                    raise KeyError("One or more values not found in axis")

                # TODO : Could use anti-join as a future optimization
                sliced_df = outdf.take(~levels_index.isin(target))
                sliced_df._index.names = self._index.names
            else:
                if errors == "raise" and not target.isin(outdf.index).all():
                    raise KeyError("One or more values not found in axis")

                sliced_df = outdf.join(
                    cudf.DataFrame(index=target), how="leftanti"
                )

            if columns is not None:
                columns = _get_host_unique(columns)
                _drop_columns(sliced_df, columns, errors)

            outdf._data = sliced_df._data
            outdf._index = sliced_df._index

        if not inplace:
            return outdf

    def _drop_column(self, name):
        """Drop a column by *name*
        """
        if name not in self._data:
            raise KeyError("column {!r} does not exist".format(name))
        del self._data[name]

    def drop_duplicates(
        self, subset=None, keep="first", inplace=False, ignore_index=False
    ):
        """
        Return DataFrame with duplicate rows removed, optionally only
        considering certain subset of columns.
        """
        outdf = super().drop_duplicates(
            subset=subset, keep=keep, ignore_index=ignore_index
        )

        return self._mimic_inplace(outdf, inplace=inplace)

    def pop(self, item):
        """Return a column and drop it from the DataFrame.
        """
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
        """
        if errors != "ignore":
            raise NotImplementedError(
                "Only errors='ignore' is currently supported"
            )

        if level:
            raise NotImplementedError(
                "Only level=False is currently supported"
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
                and type(self.index) != cudf.core.index.StringIndex
            ):
                raise NotImplementedError(
                    "Implicit conversion of index to "
                    "mixed type is not yet supported."
                )
            out = DataFrame(
                index=self.index.replace(
                    to_replace=list(index.keys()),
                    replacement=list(index.values()),
                )
            )
        else:
            out = DataFrame(index=self.index)

        if columns:
            postfix = 1
            if isinstance(columns, Mapping):
                # It is possible for DataFrames with a MultiIndex columns
                # object to have columns with the same name. The following
                # use of _cols.items and ("_1", "_2"... allows the use of
                # rename in this case
                for key, col in self._data.items():
                    if key in columns:
                        if columns[key] in out._data:
                            out_column = columns[key] + "_" + str(postfix)
                            postfix += 1
                        else:
                            out_column = columns[key]
                        out[out_column] = col
                    else:
                        out[key] = col
            elif callable(columns):
                for key, col in self._data.items():
                    out[columns(key)] = col
        else:
            out._data = self._data.copy(deep=copy)

        if inplace:
            self._data = out._data
        else:
            return out.copy(deep=copy)

    def nans_to_nulls(self):
        """
        Convert nans (if any) to nulls.
        """
        df = self.copy()
        for col in df.columns:
            df[col] = df[col].nans_to_nulls()
        return df

    def as_gpu_matrix(self, columns=None, order="F"):
        """Convert to a matrix in device memory.

        Parameters
        ----------
        columns : sequence of str
            List of a column names to be extracted.  The order is preserved.
            If None is specified, all columns are used.
        order : 'F' or 'C'
            Optional argument to determine whether to return a column major
            (Fortran) matrix or a row major (C) matrix.

        Returns
        -------
        A (nrow x ncol) numba device ndarray
        """
        if columns is None:
            columns = self._data.names

        cols = [self._data[k] for k in columns]
        ncol = len(cols)
        nrow = len(self)
        if ncol < 1:
            # This is the case for empty dataframe - construct empty cupy array
            matrix = cupy.empty(
                shape=(0, 0), dtype=np.dtype("float64"), order=order
            )
            return cuda.as_cuda_array(matrix)

        if any(
            (is_categorical_dtype(c) or np.issubdtype(c, np.dtype("object")))
            for c in cols
        ):
            raise TypeError("non-numeric data not yet supported")
        dtype = np.find_common_type(cols, [])
        for k, c in self._data.items():
            if c.has_nulls:
                errmsg = (
                    "column {!r} has null values. "
                    "hint: use .fillna() to replace null values"
                )
                raise ValueError(errmsg.format(k))
        cupy_dtype = dtype
        if np.issubdtype(cupy_dtype, np.datetime64):
            cupy_dtype = np.dtype("int64")

        if order not in ("F", "C"):
            errmsg = (
                "order parameter should be 'C' for row major or 'F' for"
                "column major GPU matrix"
            )
            raise ValueError(errmsg.format(k))

        matrix = cupy.empty(shape=(nrow, ncol), dtype=cupy_dtype, order=order)
        for colidx, inpcol in enumerate(cols):
            dense = inpcol.astype(cupy_dtype)
            matrix[:, colidx] = dense
        return cuda.as_cuda_array(matrix).view(dtype)

    def as_matrix(self, columns=None):
        """Convert to a matrix in host memory.

        Parameters
        ----------
        columns : sequence of str
            List of a column names to be extracted.  The order is preserved.
            If None is specified, all columns are used.

        Returns
        -------
        A (nrow x ncol) numpy ndarray in "F" order.
        """
        return self.as_gpu_matrix(columns=columns).copy_to_host()

    def one_hot_encoding(
        self, column, prefix, cats, prefix_sep="_", dtype="float64"
    ):
        """
        Expand a column with one-hot-encoding.

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
            the dtype for the outputs; defaults to float64.

        Returns
        -------

        a new dataframe with new columns append for each category.

        Examples
        --------
        >>> import pandas as pd
        >>> import cudf
        >>> pet_owner = [1, 2, 3, 4, 5]
        >>> pet_type = ['fish', 'dog', 'fish', 'bird', 'fish']
        >>> df = pd.DataFrame({'pet_owner': pet_owner, 'pet_type': pet_type})
        >>> df.pet_type = df.pet_type.astype('category')

        Create a column with numerically encoded category values

        >>> df['pet_codes'] = df.pet_type.cat.codes
        >>> gdf = cudf.from_pandas(df)

        Create the list of category codes to use in the encoding

        >>> codes = gdf.pet_codes.unique()
        >>> gdf.one_hot_encoding('pet_codes', 'pet_dummy', codes).head()
          pet_owner  pet_type  pet_codes  pet_dummy_0  pet_dummy_1  pet_dummy_2
        0         1      fish          2          0.0          0.0          1.0
        1         2       dog          1          0.0          1.0          0.0
        2         3      fish          2          0.0          0.0          1.0
        3         4      bird          0          1.0          0.0          0.0
        4         5      fish          2          0.0          0.0          1.0
        """
        if hasattr(cats, "to_arrow"):
            cats = cats.to_arrow().to_pylist()
        else:
            cats = pd.Series(cats, dtype="object")

        newnames = [
            prefix_sep.join([prefix, "null" if cat is None else str(cat)])
            for cat in cats
        ]
        newcols = self[column].one_hot_encoding(cats=cats, dtype=dtype)
        outdf = self.copy()
        for name, col in zip(newnames, newcols):
            outdf.insert(len(outdf._data), name, col)
        return outdf

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
        a new dataframe with a new column append for the coded values.
        """

        newname = prefix_sep.join([prefix, "labels"])
        newcol = self[column].label_encoding(
            cats=cats, dtype=dtype, na_sentinel=na_sentinel
        )
        outdf = self.copy()
        outdf.insert(len(outdf._data), newname, newcol)

        return outdf

    @annotate("ARGSORT", color="yellow", domain="cudf_python")
    def argsort(self, ascending=True, na_position="last"):
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

        Notes
        -----
        Difference from pandas:

        - Support axis='index' only.
        - Not supporting: inplace, kind
        - Ascending can be a list of bools to control per column
        """
        return self._get_sorted_inds(
            ascending=ascending, na_position=na_position
        )

    @annotate("SORT_INDEX", color="red", domain="cudf_python")
    def sort_index(
        self,
        axis=0,
        level=None,
        ascending=True,
        inplace=False,
        kind=None,
        na_position="last",
        sort_remaining=True,
        ignore_index=False,
    ):
        """Sort object by labels (along an axis).

        Parameters
        ----------
        axis : {0 or ‘index’, 1 or ‘columns’}, default 0
            The axis along which to sort. The value 0 identifies the rows,
            and 1 identifies the columns.
        level : int or level name or list of ints or list of level names
            If not None, sort on values in specified index level(s).
            This is only useful in the case of MultiIndex.
        ascending : bool, default True
            Sort ascending vs. descending.
        inplace : bool, default False
            If True, perform operation in-place.
        kind : sorting method such as `quick sort` and others.
            Not yet supported.
        na_position : {‘first’, ‘last’}, default ‘last’
            Puts NaNs at the beginning if first; last puts NaNs at the end.
        sort_remaining : bool, default True
            Not yet supported
        ignore_index : bool, default False
            if True, index will be replaced with RangeIndex.

        Returns
        -------
        DataFrame or None

        Examples
        --------
        >>> df = cudf.DataFrame(
        ... {"b":[3, 2, 1], "a":[2, 1, 3]}, index=[1, 3, 2])
        >>> df.sort_index(axis=0)
           b  a
        1  3  2
        2  1  3
        3  2  1
        >>> df.sort_index(axis=1)
           a  b
        1  2  3
        3  1  2
        2  3  1
        """
        if kind is not None:
            raise NotImplementedError("kind is not yet supported")

        if not sort_remaining:
            raise NotImplementedError(
                "sort_remaining == False is not yet supported"
            )

        if axis in (0, "index"):
            if level is not None and isinstance(self.index, cudf.MultiIndex):
                # Pandas currently don't handle na_position
                # in case of MultiIndex
                if ascending is True:
                    na_position = "first"
                else:
                    na_position = "last"

                if is_list_like(level):
                    labels = [
                        self.index._get_level_label(lvl) for lvl in level
                    ]
                else:
                    labels = [self.index._get_level_label(level)]
                inds = self.index._source_data[labels].argsort(
                    ascending=ascending, na_position=na_position
                )
            else:
                inds = self.index.argsort(
                    ascending=ascending, na_position=na_position
                )
            outdf = self.take(inds)
        else:
            labels = sorted(self._data.names, reverse=not ascending)
            outdf = self[labels]

        if ignore_index is True:
            outdf = outdf.reset_index(drop=True)
        return self._mimic_inplace(outdf, inplace=inplace)

    def sort_values(
        self,
        by,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        ignore_index=False,
    ):
        """

        Sort by the values row-wise.

        Parameters
        ----------
        by : str or list of str
            Name or list of names to sort by.
        ascending : bool or list of bool, default True
            Sort ascending vs. descending. Specify list for multiple sort
            orders. If this is a list of bools, must match the length of the
            by.
        na_position : {‘first’, ‘last’}, default ‘last’
            'first' puts nulls at the beginning, 'last' puts nulls at the end
        ignore_index : bool, default False
            If True, index will not be sorted.

        Returns
        -------
        sorted_obj : cuDF DataFrame

        Notes
        -----
        Difference from pandas:
          * Support axis='index' only.
          * Not supporting: inplace, kind

        Examples
        --------
        >>> import cudf
        >>> a = ('a', [0, 1, 2])
        >>> b = ('b', [-3, 2, 0])
        >>> df = cudf.DataFrame([a, b])
        >>> print(df.sort_values('b'))
           a  b
        0  0 -3
        2  2  0
        1  1  2
        """
        if inplace:
            raise NotImplementedError("`inplace` not currently implemented.")
        if kind != "quicksort":
            raise NotImplementedError("`kind` not currently implemented.")
        if axis != 0:
            raise NotImplementedError("`axis` not currently implemented.")

        # argsort the `by` column
        return self.take(
            self[by].argsort(ascending=ascending, na_position=na_position),
            keep_index=not ignore_index,
        )

    def nlargest(self, n, columns, keep="first"):
        """Get the rows of the DataFrame sorted by the n largest value of *columns*

        Notes
        -----
        Difference from pandas:
            - Only a single column is supported in *columns*
        """
        return self._n_largest_or_smallest("nlargest", n, columns, keep)

    def nsmallest(self, n, columns, keep="first"):
        """Get the rows of the DataFrame sorted by the n smallest value of *columns*

        Notes
        -----
        Difference from pandas:
            - Only a single column is supported in *columns*
        """
        return self._n_largest_or_smallest("nsmallest", n, columns, keep)

    def _n_largest_or_smallest(self, method, n, columns, keep):
        # Get column to operate on
        if not isinstance(columns, str):
            [column] = columns
        else:
            column = columns

        col = self[column].reset_index(drop=True)
        # Operate
        sorted_series = getattr(col, method)(n=n, keep=keep)
        df = DataFrame()
        new_positions = sorted_series.index.gpu_values
        for k in self._data.names:
            if k == column:
                df[k] = sorted_series
            else:
                df[k] = self[k].reset_index(drop=True).take(new_positions)
        return df.set_index(self.index.take(new_positions))

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
        # Cython renames the columns to the range [0...ncols]
        result = self.__class__._from_table(libcudf.transpose.transpose(self))
        # Set the old column names as the new index
        result._index = as_index(index)
        # Set the old index as the new column names
        result.columns = columns
        return result

    @property
    def T(self):
        """
        Transpose index and columns.

        Reflect the DataFrame over its main diagonal by writing rows
        as columns and vice-versa. The property T is an accessor to
        the method transpose().

        Returns
        -------
        out : DataFrame
            The transposed DataFrame.
        """

        return self.transpose()

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
        type="",
        method="hash",
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
        how : {‘left’, ‘outer’, ‘inner’}, default ‘inner’
            Type of merge to be performed.

            - left : use only keys from left frame, similar to a SQL left
              outer join.
            - right : not supported.
            - outer : use union of keys from both frames, similar to a SQL
              full outer join.
            - inner: use intersection of keys from both frames, similar to
              a SQL inner join.
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
        method : {‘hash’, ‘sort’}, default ‘hash’
            The implementation method to be used for the operation.

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

        if type != "":
            warnings.warn(
                'type="' + type + '" parameter is deprecated.'
                'Use method="' + type + '" instead.',
                DeprecationWarning,
            )
            method = type

        lhs = self.copy(deep=False)
        rhs = right.copy(deep=False)

        # Compute merge
        gdf_result = super(DataFrame, lhs)._merge(
            rhs,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            how=how,
            sort=sort,
            lsuffix=lsuffix,
            rsuffix=rsuffix,
            method=method,
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
        type="",
        method="hash",
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

        Notes
        -----
        Difference from pandas:

        - *other* must be a single DataFrame for now.
        - *on* is not supported yet due to lack of multi-index support.
        """
        # Outer joins still use the old implementation
        if type != "":
            warnings.warn(
                'type="' + type + '" parameter is deprecated.'
                'Use method="' + type + '" instead.',
                DeprecationWarning,
            )
            method = type

        lhs = self
        rhs = other

        df = lhs.merge(
            rhs,
            left_index=True,
            right_index=True,
            how=how,
            suffixes=(lsuffix, rsuffix),
            method=method,
            sort=sort,
        )
        df.index.name = (
            None if lhs.index.name != rhs.index.name else lhs.index.name
        )
        return df

    @copy_docstring(DataFrameGroupBy.__init__)
    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=True,
        group_keys=True,
        squeeze=False,
        observed=False,
        dropna=True,
        method=None,
    ):
        if axis not in (0, "index"):
            raise NotImplementedError("axis parameter is not yet implemented")

        if squeeze is not False:
            raise NotImplementedError(
                "squeeze parameter is not yet implemented"
            )

        if observed is not False:
            raise NotImplementedError(
                "observed parameter is not yet implemented"
            )

        if group_keys is not True:
            raise NotImplementedError(
                "The group_keys keyword is not yet implemented"
            )
        if by is None and level is None:
            raise TypeError(
                "groupby() requires either by or level to be" "specified."
            )

        if method is not None:
            warnings.warn(
                "The 'method' argument is deprecated and will be unused",
                DeprecationWarning,
            )
        return DataFrameGroupBy(
            self,
            by=by,
            level=level,
            as_index=as_index,
            dropna=dropna,
            sort=sort,
        )

    @copy_docstring(Rolling)
    def rolling(
        self, window, min_periods=None, center=False, axis=0, win_type=None
    ):
        return Rolling(
            self,
            window,
            min_periods=min_periods,
            center=center,
            axis=axis,
            win_type=win_type,
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
        >>> print(df.query(expr))
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
        >>> print(df.query('datetimes==@search_date'))
                        datetimes
        1 2018-10-08T00:00:00.000

        Using local_dict:

        >>> import numpy as np
        >>> import datetime
        >>> df = cudf.DataFrame()
        >>> data = np.array(['2018-10-07', '2018-10-08'], dtype='datetime64')
        >>> df['datetimes'] = data
        >>> search_date2 = datetime.datetime.strptime('2018-10-08', '%Y-%m-%d')
        >>> print(df.query('datetimes==@search_date',
        >>>         local_dict={'search_date':search_date2}))
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
                    "local_dict type: expected dict but found {!r}".format(
                        type(local_dict)
                    )
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

    def hash_columns(self, columns=None):
        """Hash the given *columns* and return a new device array

        Parameters
        ----------
        columns : sequence of str; optional
            Sequence of column names. If columns is *None* (unspecified),
            all columns in the frame are used.
        """
        if columns is None:
            table_to_hash = self
        else:
            cols = [self[k]._column for k in columns]
            table_to_hash = Frame(data=OrderedColumnDict(zip(columns, cols)))

        return Series(table_to_hash._hash()).values

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
        outdf, offsets = self._hash_partition(key_indices, nparts, keep_index)
        # Slice into partition
        return [outdf[s:e] for s, e in zip(offsets, offsets[1:] + [None])]

    def replace(
        self,
        to_replace=None,
        value=None,
        inplace=False,
        limit=None,
        regex=False,
        method=None,
    ):
        """
        Replace values given in *to_replace* with *replacement*.

        Parameters
        ----------
        to_replace : numeric, str, list-like or dict
            Value(s) to replace.

            * numeric or str:

                - values equal to *to_replace* will be replaced
                  with *replacement*

            * list of numeric or str:

                - If *replacement* is also list-like,
                  *to_replace* and *replacement* must be of same length.

            * dict:

                - Dicts can be used to replace different values in different
                  columns. For example, `{'a': 1, 'z': 2}` specifies that the
                  value 1 in column `a` and the value 2 in column `z` should be
                  replaced with replacement*.
        value : numeric, str, list-like, or dict
            Value(s) to replace `to_replace` with. If a dict is provided, then
            its keys must match the keys in *to_replace*, and corresponding
            values must be compatible (e.g., if they are lists, then they must
            match in length).
        inplace : bool, default False
            If True, in place.

        Returns
        -------
        result : DataFrame
            DataFrame after replacement.

        Examples
        --------
        >>> import cudf
        >>> gdf = cudf.DataFrame()
        >>> gdf['id']= [0, 1, 2, -1, 4, -1, 6]
        >>> gdf['id']= gdf['id'].replace(-1, None)
        >>> gdf
             id
        0     0
        1     1
        2     2
        3  null
        4     4
        5  null
        6     6

        Notes
        -----
        Parameters that are currently not supported are: `limit`, `regex`,
        `method`
        """
        if limit is not None:
            raise NotImplementedError("limit parameter is not implemented yet")

        if regex:
            raise NotImplementedError("regex parameter is not implemented yet")

        if method not in ("pad", None):
            raise NotImplementedError(
                "method parameter is not implemented yet"
            )

        outdf = super().replace(to_replace=to_replace, replacement=value)

        return self._mimic_inplace(outdf, inplace=inplace)

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
            names = OrderedDict.fromkeys(
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

    def to_pandas(self, **kwargs):
        """
        Convert to a Pandas DataFrame.

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
        """

        out_data = {}
        out_index = self.index.to_pandas()

        if not isinstance(self.columns, pd.Index):
            out_columns = self.columns.to_pandas()
        else:
            out_columns = self.columns

        for i, col_key in enumerate(self._data):
            out_data[i] = self._data[col_key].to_pandas(index=out_index)

        if isinstance(self.columns, Index):
            out_columns = self.columns.to_pandas()
            if isinstance(self.columns, cudf.core.multiindex.MultiIndex):
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
        if isinstance(dataframe.index, pd.MultiIndex):
            index = cudf.from_pandas(dataframe.index, nan_as_null=nan_as_null)
        else:
            index = dataframe.index
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
                if isinstance(self.index, cudf.MultiIndex):
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
            self,
            out.schema.names,
            [self.index],
            index_descr,
            preserve_index,
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
            ret["index"] = self.index.to_array()
        for col in self._data.names:
            ret[col] = self[col].to_array()
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
                "records dimension expected 1 or 2 but found {!r}".format(
                    data.ndim
                )
            )

        num_cols = len(data[0])

        if columns is None and data.dtype.names is None:
            names = [i for i in range(num_cols)]

        elif data.dtype.names is not None:
            names = data.dtype.names

        else:
            if len(columns) != num_cols:
                msg = "columns length expected {!r} but found {!r}"
                raise ValueError(msg.format(num_cols, len(columns)))
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
            num_cols = len(data[0])
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

    @classmethod
    def from_gpu_matrix(
        self, data, index=None, columns=None, nan_as_null=False
    ):
        """Convert from a numba gpu ndarray.

        Parameters
        ----------
        data : numba gpu ndarray
        index : str, Index
            The name of the index column in `data` or an Index itself.
            If None, the default index is used.
        columns : list of str
            List of column names to include.

        Returns
        -------
        DataFrame
        """
        warnings.warn(
            "DataFrame.from_gpu_matrix will be removed in 0.16. "
            "Please use cudf.DataFrame() to create a DataFrame "
            "out of a gpu matrix",
            DeprecationWarning,
            stacklevel=2,
        )

        if data.ndim != 2:
            raise ValueError(
                f"matrix dimension expected 2 but found {data.ndim}"
            )

        if columns is None:
            names = [i for i in range(data.shape[1])]
        else:
            if len(columns) != data.shape[1]:
                raise ValueError(
                    f"columns length expected {data.shape[1]} but "
                    f"found {len(columns)}"
                )
            names = columns

        if (
            index is not None
            and not isinstance(index, (str, int))
            and len(index) != data.shape[0]
        ):
            raise ValueError(
                f"index length expected {data.shape[0]} but found {len(index)}"
            )

        df = DataFrame()
        data = cupy.asfortranarray(cupy.asarray(data))
        for i, k in enumerate(names):
            df._data[k] = as_column(data[:, i], nan_as_null=nan_as_null)

        if index is not None:
            if isinstance(index, (str, int)):
                index = as_index(df[index])
            else:
                index = as_index(index)
        else:
            index = RangeIndex(start=0, stop=len(data))
        df._index = index

        return df

    def to_gpu_matrix(self):
        """Convert to a numba gpu ndarray

        Returns
        -------
        numba gpu ndarray
        """
        warnings.warn(
            "The to_gpu_matrix method will be deprecated"
            "in the future. use as_gpu_matrix instead.",
            DeprecationWarning,
        )
        return self.as_gpu_matrix()

    @classmethod
    def _from_columns(cls, cols, index=None, columns=None):
        """
        Construct a DataFrame from a list of Columns
        """
        if columns is not None:
            data = dict(zip(columns, cols))
        else:
            data = dict(enumerate(cols))

        return cls(data=data, index=index,)

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
        numeric_only : boolean
            numeric_only is a NON-FUNCTIONAL parameter
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

        DataFrame
        """
        if axis not in (0, None):
            raise NotImplementedError("axis is not implemented yet")

        if not numeric_only:
            raise NotImplementedError("numeric_only is not implemented yet")
        if columns is None:
            columns = self._data.names

        result = DataFrame()

        for k in self._data.names:

            if k in columns:
                res = self[k].quantile(
                    q,
                    interpolation=interpolation,
                    exact=exact,
                    quant_index=False,
                )
                if not isinstance(res, numbers.Number) and len(res) == 0:
                    res = column.column_empty_like(
                        q, dtype="float64", masked=True, newsize=len(q)
                    )
                result[k] = column.as_column(res)

        if isinstance(q, numbers.Number):
            result = result.fillna(np.nan)
            result = result.iloc[0]
            result.index = as_index(self.columns)
            result.name = q
            return result
        else:
            q = list(map(float, q))
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

            for col in self._data.names:
                if isinstance(
                    self[col]._column, cudf.core.column.CategoricalColumn
                ) and isinstance(
                    values._column, cudf.core.column.CategoricalColumn
                ):
                    res = self._data[col].binary_operator("eq", values._column)
                    result[col] = res
                elif (
                    isinstance(
                        self[col]._column, cudf.core.column.CategoricalColumn
                    )
                    or np.issubdtype(self[col].dtype, np.dtype("object"))
                ) or (
                    isinstance(
                        values._column, cudf.core.column.CategoricalColumn
                    )
                    or np.issubdtype(values.dtype, np.dtype("object"))
                ):
                    result[col] = utils.scalar_broadcast_to(False, len(self))
                else:
                    result[col] = self._data[col].binary_operator(
                        "eq", values._column
                    )

            result.index = self.index
            return result
        elif isinstance(values, DataFrame):
            values = values.reindex(self.index)

            result = DataFrame()
            for col in self._data.names:
                if col in values.columns:
                    result[col] = self._data[col].binary_operator(
                        "eq", values[col]._column
                    )
                else:
                    result[col] = utils.scalar_broadcast_to(False, len(self))
            result.index = self.index
            return result
        else:
            if not is_list_like(values):
                raise TypeError(
                    "only list-like or dict-like objects are "
                    "allowed to be passed to DataFrame.isin(), "
                    "you passed a "
                    "{0!r}".format(type(values).__name__)
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
        """Prepare a DataFrame for CuPy-based row-wise operations.
        """

        if method not in _cupy_nan_methods_map and any(
            col.nullable for col in self._columns
        ):
            msg = (
                f"Row-wise operations to calculate '{method}' is not "
                f"currently support columns with null values. "
                f"Consider removing them with .dropna() "
                f"or using .fillna()."
            )
            raise ValueError(msg)

        filtered = self.select_dtypes(include=[np.number, np.bool])
        common_dtype = np.find_common_type(filtered.dtypes, [])
        if filtered._num_columns < self._num_columns:
            msg = (
                "Row-wise operations currently only support int, float "
                "and bool dtypes. Non numeric columns are ignored."
            )
            warnings.warn(msg)

        if not skipna and any(col.nullable for col in filtered._columns):
            mask = cudf.DataFrame(
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

        coerced = filtered.astype(common_dtype)
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
        return self._apply_support_method(
            "count",
            axis=axis,
            level=level,
            numeric_only=numeric_only,
            **kwargs,
        )

    def min(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs,
    ):
        """
        Return the minimum of the values in the DataFrame.

        Parameters
        ----------
        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        level: int or level name, default None
            If the axis is a MultiIndex (hierarchical), count along a
            particular level, collapsing into a Series.
        numeric_only: bool, default None
            Include only float, int, boolean columns. If None, will attempt to
            use everything, then use only numeric data.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are `level`, `numeric_only`.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.min()
        a    1
        b    7
        dtype: int64
        """
        return self._apply_support_method(
            "min",
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs,
        )

    def max(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs,
    ):
        """
        Return the maximum of the values in the DataFrame.

        Parameters
        ----------
        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        level: int or level name, default None
            If the axis is a MultiIndex (hierarchical), count along a
            particular level, collapsing into a Series.
        numeric_only: bool, default None
            Include only float, int, boolean columns. If None, will attempt to
            use everything, then use only numeric data.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are `level`, `numeric_only`.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.max()
        a     4
        b    10
        dtype: int64
        """
        return self._apply_support_method(
            "max",
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs,
        )

    def sum(
        self,
        axis=None,
        skipna=None,
        dtype=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):
        """
        Return sum of the values in the DataFrame.

        Parameters
        ----------

        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        dtype: data type
            Data type to cast the result to.
        min_count: int, default 0
            The required number of valid values to perform the operation.
            If fewer than min_count non-NA values are present the result
            will be NA.

            The default being 0. This means the sum of an all-NA or empty
            Series is 0, and the product of an all-NA or empty Series is 1.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are `level`, `numeric_only`.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.sum()
        a    10
        b    34
        dtype: int64
        """
        return self._apply_support_method(
            "sum",
            axis=axis,
            skipna=skipna,
            dtype=dtype,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    def product(
        self,
        axis=None,
        skipna=None,
        dtype=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):
        """
        Return product of the values in the DataFrame.

        Parameters
        ----------

        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        dtype: data type
            Data type to cast the result to.
        min_count: int, default 0
            The required number of valid values to perform the operation.
            If fewer than min_count non-NA values are present the result
            will be NA.

            The default being 0. This means the sum of an all-NA or empty
            Series is 0, and the product of an all-NA or empty Series is 1.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are level`, `numeric_only`.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.product()
        a      24
        b    5040
        dtype: int64
        """
        return self._apply_support_method(
            "prod",
            axis=axis,
            skipna=skipna,
            dtype=dtype,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    def prod(
        self,
        axis=None,
        skipna=None,
        dtype=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs,
    ):
        """
        Return product of the values in the DataFrame.

        Parameters
        ----------

        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values when computing the result.
        dtype: data type
            Data type to cast the result to.
        min_count: int, default 0
            The required number of valid values to perform the operation.
            If fewer than min_count non-NA values are present the result
            will be NA.

            The default being 0. This means the sum of an all-NA or empty
            Series is 0, and the product of an all-NA or empty Series is 1.

        Returns
        -------
        scalar

        Notes
        -----
        Parameters currently not supported are `level`, `numeric_only`.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.prod()
        a      24
        b    5040
        dtype: int64
        """
        return self.product(
            axis=axis,
            skipna=skipna,
            dtype=dtype,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs,
        )

    def cummin(self, axis=None, skipna=True, *args, **kwargs):
        """
        Return cumulative minimum of the DataFrame.

        Parameters
        ----------

        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA,
            the result will be NA.

        Returns
        -------
        DataFrame

        Notes
        -----
        Parameters currently not supported is `axis`

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.cummin()
           a  b
        0  1  7
        1  1  7
        2  1  7
        3  1  7
        """
        return self._apply_support_method(
            "cummin", axis=axis, skipna=skipna, *args, **kwargs
        )

    def cummax(self, axis=None, skipna=True, *args, **kwargs):
        """
        Return cumulative maximum of the DataFrame.

        Parameters
        ----------

        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA,
            the result will be NA.

        Returns
        -------
        DataFrame

        Notes
        -----
        Parameters currently not supported is `axis`

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.cummax()
           a   b
        0  1   7
        1  2   8
        2  3   9
        3  4  10
        """
        return self._apply_support_method(
            "cummax", axis=axis, skipna=skipna, *args, **kwargs
        )

    def cumsum(self, axis=None, skipna=True, *args, **kwargs):
        """
        Return cumulative sum of the DataFrame.

        Parameters
        ----------

        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA,
            the result will be NA.


        Returns
        -------
        DataFrame

        Notes
        -----
        Parameters currently not supported is `axis`

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> s.cumsum()
            a   b
        0   1   7
        1   3  15
        2   6  24
        3  10  34
        """
        return self._apply_support_method(
            "cumsum", axis=axis, skipna=skipna, *args, **kwargs
        )

    def cumprod(self, axis=None, skipna=True, *args, **kwargs):
        """
        Return cumulative product of the DataFrame.

        Parameters
        ----------

        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA,
            the result will be NA.

        Returns
        -------
        DataFrame

        Notes
        -----
        Parameters currently not supported is `axis`

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> s.cumprod()
            a     b
        0   1     7
        1   2    56
        2   6   504
        3  24  5040
        """
        return self._apply_support_method(
            "cumprod", axis=axis, skipna=skipna, *args, **kwargs
        )

    def mean(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs
    ):
        """
        Return the mean of the values for the requested axis.

        Parameters
        ----------
        axis : {0 or 'index', 1 or 'columns'}
            Axis for the function to be applied on.
        skipna : bool, default True
            Exclude NA/null values when computing the result.
        level : int or level name, default None
            If the axis is a MultiIndex (hierarchical), count along a
            particular level, collapsing into a Series.
        numeric_only : bool, default None
            Include only float, int, boolean columns. If None, will attempt to
            use everything, then use only numeric data. Not implemented for
            Series.
        **kwargs
            Additional keyword arguments to be passed to the function.

        Returns
        -------
        mean : Series or DataFrame (if level specified)

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.mean()
        a    2.5
        b    8.5
        dtype: float64
        """
        return self._apply_support_method(
            "mean",
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs,
        )

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
        cudf.core.series.Series.mode : Return the highest frequency value
            in a Series.
        cudf.core.series.Series.value_counts : Return the counts of values
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
            df = DataFrame(index=self.index)
            return df

        df = cudf.concat(mode_results, axis=1)
        if isinstance(df, Series):
            df = df.to_frame()

        df.columns = data_df.columns

        return df

    def std(
        self,
        axis=None,
        skipna=None,
        level=None,
        ddof=1,
        numeric_only=None,
        **kwargs,
    ):
        """
        Return sample standard deviation of the DataFrame.

        Normalized by N-1 by default. This can be changed using
        the `ddof` argument

        Parameters
        ----------

        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof: int, default 1
            Delta Degrees of Freedom. The divisor used in calculations
            is N - ddof, where N represents the number of elements.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are `level` and
        `numeric_only`

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.std()
        a    1.290994
        b    1.290994
        dtype: float64
        """

        return self._apply_support_method(
            "std",
            axis=axis,
            skipna=skipna,
            level=level,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    def var(
        self,
        axis=None,
        skipna=None,
        level=None,
        ddof=1,
        numeric_only=None,
        **kwargs,
    ):
        """
        Return unbiased variance of the DataFrame.

        Normalized by N-1 by default. This can be changed using the
        ddof argument

        Parameters
        ----------

        axis: {index (0), columns(1)}
            Axis for the function to be applied on.
        skipna: bool, default True
            Exclude NA/null values. If an entire row/column is NA, the result
            will be NA.
        ddof: int, default 1
            Delta Degrees of Freedom. The divisor used in calculations is
            N - ddof, where N represents the number of elements.

        Returns
        -------
        scalar

        Notes
        -----
        Parameters currently not supported are `level` and
        `numeric_only`

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.var()
        a    1.666667
        b    1.666667
        dtype: float64
        """
        return self._apply_support_method(
            "var",
            axis=axis,
            skipna=skipna,
            level=level,
            ddof=ddof,
            numeric_only=numeric_only,
            **kwargs,
        )

    def kurtosis(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs
    ):
        """
        Return Fisher's unbiased kurtosis of a sample.

        Kurtosis obtained using Fisher’s definition of
        kurtosis (kurtosis of normal == 0.0). Normalized by N-1.

        Parameters
        ----------

        skipna: bool, default True
            Exclude NA/null values when computing the result.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are `axis`, `level` and
        `numeric_only`

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [1, 2, 3, 4], 'b': [7, 8, 9, 10]})
        >>> df.kurt()
        a   -1.2
        b   -1.2
        dtype: float64
        """
        if numeric_only not in (None, True):
            msg = "Kurtosis only supports int, float, and bool dtypes."
            raise NotImplementedError(msg)

        self = self.select_dtypes(include=[np.number, np.bool])
        return self._apply_support_method(
            "kurtosis",
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs,
        )

    # Alias for kurtosis.
    kurt = kurtosis

    def skew(
        self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs
    ):
        """
        Return unbiased Fisher-Pearson skew of a sample.

        Parameters
        ----------
        skipna: bool, default True
            Exclude NA/null values when computing the result.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are `axis`, `level` and
        `numeric_only`

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [3, 2, 3, 4], 'b': [7, 8, 10, 10]})
        >>> df.skew()
        a    0.00000
        b   -0.37037
        dtype: float64
        """
        if numeric_only not in (None, True):
            msg = "Skew only supports int, float, and bool dtypes."
            raise NotImplementedError(msg)

        self = self.select_dtypes(include=[np.number, np.bool])
        return self._apply_support_method(
            "skew",
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs,
        )

    def all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """
        Return whether all elements are True in DataFrame.

        Parameters
        ----------

        skipna: bool, default True
            Exclude NA/null values. If the entire row/column is NA and
            skipna is True, then the result will be True, as for an
            empty row/column.
            If skipna is False, then NA are treated as True, because
            these are not equal to zero.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are `axis`, `bool_only`, `level`.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [3, 2, 3, 4], 'b': [7, 0, 10, 10]})
        >>> df.all()
        a     True
        b    False
        dtype: bool
        """
        if bool_only:
            return self.select_dtypes(include="bool")._apply_support_method(
                "all",
                axis=axis,
                bool_only=bool_only,
                skipna=skipna,
                level=level,
                **kwargs,
            )
        return self._apply_support_method(
            "all",
            axis=axis,
            bool_only=bool_only,
            skipna=skipna,
            level=level,
            **kwargs,
        )

    def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """
        Return whether any elements is True in DataFrame.

        Parameters
        ----------

        skipna: bool, default True
            Exclude NA/null values. If the entire row/column is NA and
            skipna is True, then the result will be False, as for an
            empty row/column.
            If skipna is False, then NA are treated as True, because
            these are not equal to zero.

        Returns
        -------
        Series

        Notes
        -----
        Parameters currently not supported are `axis`, `bool_only`, `level`.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame({'a': [3, 2, 3, 4], 'b': [7, 0, 10, 10]})
        >>> df.any()
        a    True
        b    True
        dtype: bool
        """
        if bool_only:
            return self.select_dtypes(include="bool")._apply_support_method(
                "any",
                axis=axis,
                bool_only=bool_only,
                skipna=skipna,
                level=level,
                **kwargs,
            )
        return self._apply_support_method(
            "any",
            axis=axis,
            bool_only=bool_only,
            skipna=skipna,
            level=level,
            **kwargs,
        )

    def _apply_support_method(self, method, axis=0, *args, **kwargs):
        assert axis in (None, 0, 1)

        if axis in (None, 0):
            result = [
                getattr(self[col], method)(*args, **kwargs)
                for col in self._data.names
            ]

            if isinstance(result[0], Series):
                support_result = result
                result = DataFrame(index=support_result[0].index)
                for idx, col in enumerate(self._data.names):
                    result[col] = support_result[idx]
            else:
                result = Series(result)
                result = result.set_index(self._data.names)
            return result

        elif axis == 1:
            # for dask metadata compatibility
            skipna = kwargs.pop("skipna", None)
            if method not in _cupy_nan_methods_map and skipna not in (
                None,
                True,
                1,
            ):
                raise NotImplementedError(
                    f"Row-wise operation to calculate '{method}'"
                    f" currently do not support `skipna=False`."
                )

            level = kwargs.pop("level", None)
            if level not in (None,):
                msg = "Row-wise operations currently do not "
                "support `level`."
                raise NotImplementedError(msg)

            numeric_only = kwargs.pop("numeric_only", None)
            if numeric_only not in (None, True):
                msg = "Row-wise operations currently do not "
                "support `numeric_only=False`."
                raise NotImplementedError(msg)

            min_count = kwargs.pop("min_count", None)
            if min_count not in (None, 0):
                msg = "Row-wise operations currently do not "
                "support `min_count`."
                raise NotImplementedError(msg)

            bool_only = kwargs.pop("bool_only", None)
            if bool_only not in (None, True):
                msg = "Row-wise operations currently do not "
                "support `bool_only`."
                raise NotImplementedError(msg)

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
                        )
                        .fillna(np.nan)
                    )
            arr = cupy.asarray(prepared.as_gpu_matrix())

            if skipna is not False and method in _cupy_nan_methods_map:
                method = _cupy_nan_methods_map[method]

            result = getattr(cupy, method)(arr, axis=1, **kwargs)

            if result.ndim == 1:
                result = column.as_column(result)
                if mask is not None:
                    result = result.set_mask(
                        cudf._lib.transform.bools_to_mask(mask._column)
                    )
                return Series(
                    result,
                    index=self.index,
                    dtype=common_dtype
                    if method
                    in {
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
                    else None,
                )
            else:
                result_df = DataFrame.from_gpu_matrix(result).set_index(
                    self.index
                )
                result_df.columns = prepared.columns
                return result_df

    def _columns_view(self, columns):
        """
        Return a subset of the DataFrame's columns as a view.
        """
        result_columns = OrderedDict({})
        for col in columns:
            result_columns[col] = self._data[col]
        return DataFrame(result_columns, index=self.index)

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
                "include and exclude overlap on {inc_ex}".format(
                    inc_ex=(include & exclude)
                )
            )

        # include all subtypes
        include_subtypes = set()
        for dtype in self.dtypes:
            for i_dtype in include:
                # category handling
                if is_categorical_dtype(i_dtype):
                    include_subtypes.add(i_dtype)
                elif issubclass(dtype.type, i_dtype):
                    include_subtypes.add(dtype.type)

        # exclude all subtypes
        exclude_subtypes = set()
        for dtype in self.dtypes:
            for e_dtype in exclude:
                # category handling
                if is_categorical_dtype(e_dtype):
                    exclude_subtypes.add(e_dtype)
                elif issubclass(dtype.type, e_dtype):
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

    @ioutils.doc_to_json()
    def to_json(self, path_or_buf=None, *args, **kwargs):
        """{docstring}"""
        from cudf.io import json as json

        return json.to_json(self, path_or_buf=path_or_buf, *args, **kwargs)

    @ioutils.doc_to_hdf()
    def to_hdf(self, path_or_buf, key, *args, **kwargs):
        """{docstring}"""
        from cudf.io import hdf as hdf

        hdf.to_hdf(path_or_buf, key, self, *args, **kwargs)

    @ioutils.doc_to_dlpack()
    def to_dlpack(self):
        """{docstring}"""
        from cudf.io import dlpack as dlpack

        return dlpack.to_dlpack(self)

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
        new_index = cudf.core.multiindex.MultiIndex.from_frame(
            DataFrame(dict(zip(range(0, len(new_index)), new_index)))
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
        df = DataFrame.from_gpu_matrix(cupy.asfortranarray(cov)).set_index(
            self.columns
        )
        df.columns = self.columns
        return df

    def corr(self):
        """Compute the correlation matrix of a DataFrame.
        """
        corr = cupy.corrcoef(self.values, rowvar=False)
        df = DataFrame.from_gpu_matrix(cupy.asfortranarray(corr)).set_index(
            self.columns
        )
        df.columns = self.columns
        return df

    def to_dict(self, orient="dict", into=dict):
        raise TypeError(
            "cuDF does not support conversion to host memory "
            "via `to_dict()` method. Consider using "
            "`.to_pandas().to_dict()` to construct a Python dictionary."
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
        cudf.core.reshape.concat : General function to concatenate DataFrame or
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
            elif not isinstance(other[0], cudf.DataFrame):
                other = cudf.DataFrame(other)
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

    def equals(self, other):
        if not isinstance(other, DataFrame):
            return False
        for self_name, other_name in zip(self._data.names, other._data.names):
            if self_name != other_name:
                return False
        return super().equals(other)

    _accessors = set()


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
    MultiIndex(levels=[0    1
    1    3
    2    4
    3    5
    dtype: int64, 0    1
    1    2
    2    5
    dtype: int64],
    codes=   x  y
    0  0  0
    1  0  2
    2  1  1
    3  2  1
    4  3  0)
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
        return cudf.MultiIndex.from_pandas(obj, nan_as_null=nan_as_null)
    elif isinstance(obj, pd.RangeIndex):
        if obj._step and obj._step != 1:
            raise ValueError("cudf RangeIndex requires step == 1")
        return cudf.core.index.RangeIndex(
            obj._start, stop=obj._stop, name=obj.name
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


def _setitem_with_dataframe(input_df, replace_df, input_cols=None, mask=None):
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
            and not isinstance(df.index, cudf.MultiIndex)
        ):
            return df.index._data.columns[0]
        return df.index._data[col]


def _get_union_of_indices(indexes):
    if len(indexes) == 1:
        return indexes[0]
    else:
        merged_index = cudf.core.Index._concat(indexes)
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


def _drop_columns(df, columns, errors):
    for c in columns:
        try:
            df._drop_column(c)
        except KeyError as e:
            if errors == "ignore":
                pass
            else:
                raise e


def _get_host_unique(array):
    if isinstance(
        array, (cudf.Series, cudf.Index, cudf.core.column.ColumnBase)
    ):
        return array.unique.to_pandas()
    elif isinstance(array, (str, numbers.Number)):
        return [array]
    else:
        return set(array)
