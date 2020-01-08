# Copyright (c) 2018-2019, NVIDIA CORPORATION.

from __future__ import division, print_function

import inspect
import itertools
import logging
import numbers
import pickle
import uuid
import warnings
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from types import GeneratorType

import cupy
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.types import is_dict_like

import rmm

import cudf
import cudf._lib as libcudf
from cudf.core import column
from cudf.core._sort import get_sorted_inds
from cudf.core.column import CategoricalColumn, StringColumn, as_column
from cudf.core.index import Index, RangeIndex, as_index
from cudf.core.indexing import _DataFrameIlocIndexer, _DataFrameLocIndexer
from cudf.core.series import Series
from cudf.core.table import Table
from cudf.core.window import Rolling
from cudf.utils import applyutils, cudautils, ioutils, queryutils, utils
from cudf.utils.docutils import copy_docstring
from cudf.utils.dtypes import (
    cudf_dtype_from_pydata_dtype,
    is_categorical_dtype,
    is_datetime_dtype,
    is_list_like,
    is_scalar,
    is_string_dtype,
)


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


class DataFrame(Table):
    """
    A GPU Dataframe object.

    Parameters
    ----------
    data : data-type to coerce. Infers date format if to date.

    Examples
    --------

    Build dataframe with `__setitem__`:

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
    >>>   'id': np.arange(n),
    >>>   'datetimes', np.array([(t0+ timedelta(seconds=x)) for x in range(n)])
    >>> })
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
        (5, "cats", "jump", np.nan),
        (2, "dogs", "dig", 7.5),
        (3, "cows", "moo", -2.1, "occasionally"),
    ])
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

    def __init__(self, data=None, index=None, columns=None, dtype=None):
        super().__init__()

        self._columns_name = None

        if isinstance(columns, cudf.MultiIndex):
            self.multi_cols = columns
            columns = RangeIndex(len(columns))

        if data is None:
            if index is None:
                self._index = RangeIndex(0)
            else:
                self._index = as_index(index)
            if columns is not None:
                self._data = OrderedDict.fromkeys(
                    columns,
                    column.column_empty(
                        len(self), dtype="object", masked=True
                    ),
                )
        else:
            if is_list_like(data):
                if len(data) > 0 and is_scalar(data[0]):
                    data = [data]
                self._init_from_list_like(data, index=index, columns=columns)

            else:
                if not is_dict_like(data):
                    raise TypeError("data must be list or dict-like")

                self._init_from_dict_like(data, index=index, columns=columns)

        if dtype:
            self._data = self.astype(dtype)._data

        # allows Pandas-like __setattr__ functionality: `df.x = column`, etc.
        self._allow_setattr_to_setitem = True

    def _init_from_list_like(self, data, index=None, columns=None):
        if index is None:
            index = RangeIndex(start=0, stop=len(data))
        else:
            index = as_index(index)
        self._index = as_index(index)
        data = list(itertools.zip_longest(*data))

        if columns is None:
            columns = range(len(data))

        for col_name, col in enumerate(data):
            self._data[col_name] = column.as_column(col)

        if not hasattr(self, "multi_cols"):
            self.columns = columns

    def _init_from_dict_like(self, data, index=None, columns=None):
        data = data.copy()
        num_rows = 0

        if columns is not None:
            # remove all entries in `data` that are
            # not in `columns`
            keys = [key for key in data.keys() if key in columns]
            data = {key: data[key] for key in keys}

            if keys:
                # if keys is non-empty,
                # add null columns for all values
                # in `columns` that don't exist in `keys`:
                extra_cols = [col for col in columns if col not in data.keys()]
                data.update({key: None for key in extra_cols})

        if index is None:
            for i, col_name in enumerate(data):
                if is_scalar(data[col_name]):
                    num_rows = num_rows or 1
                else:
                    data[col_name] = column.as_column(data[col_name])
                    num_rows = data[col_name].size
            self._index = RangeIndex(0, num_rows)
        else:
            self._index = as_index(index)

        for (i, col_name) in enumerate(data):
            self.insert(i, col_name, data[col_name])

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
        header["type"] = pickle.dumps(type(self))
        header["index"], index_frames = self._index.serialize()
        header["index_frame_count"] = len(index_frames)
        frames.extend(index_frames)

        # Use the column directly to avoid duplicating the index
        # need to pickle column names to handle numpy integer columns
        header["column_names"] = pickle.dumps(tuple(self._data.keys()))
        column_header, column_frames = column.serialize_columns(self._columns)
        header["columns"] = column_header
        frames.extend(column_frames)

        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        # Reconstruct the index
        index_frames = frames[: header["index_frame_count"]]

        idx_typ = pickle.loads(header["index"]["type"])
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
            [x.dtype for x in self._data.values()], index=self._data.keys()
        )

    @property
    def shape(self):
        """Returns a tuple representing the dimensionality of the DataFrame.
        """
        return len(self._index), len(self._data)

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
        if getattr(self, "_allow_setattr_to_setitem", False):
            # if an attribute already exists, set it.
            try:
                object.__getattribute__(self, key)
                object.__setattr__(self, key, col)
                return
            except AttributeError:
                pass

            # if a column already exists, set it.
            try:
                self[key]  # __getitem__ to verify key exists
                self[key] = col
                return
            except KeyError:
                pass

            warnings.warn(
                "Columns may not be added to a DataFrame using a new "
                + "attribute name. A new attribute will be created: '%s'"
                % key,
                UserWarning,
            )

        object.__setattr__(self, key, col)

    def __getattr__(self, key):
        if key != "_data" and key in self._data:
            return self[key]

        raise AttributeError("'DataFrame' object has no attribute %r" % key)

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
        if isinstance(
            self.columns, cudf.core.multiindex.MultiIndex
        ) and isinstance(arg, tuple):
            return self.columns._get_column_major(self, arg)
        if is_scalar(arg) or isinstance(arg, tuple):
            s = cudf.Series(self._data[arg], name=arg, index=self.index)
            return s
        elif isinstance(arg, slice):
            df = DataFrame(index=self.index[arg])
            for k, col in self._data.items():
                df[k] = col[arg]
            return df
        elif isinstance(
            arg, (list, np.ndarray, pd.Series, Series, Index, pd.Index)
        ):
            mask = arg
            if isinstance(mask, list):
                mask = np.array(mask)
            df = DataFrame()

            if mask.dtype == "bool":
                # New df-wide index
                index = self.index.take(mask)
                for col in self._data:
                    df[col] = self._data[col][arg]
                df = df.set_index(index)
            else:
                if len(arg) == 0:
                    df.index = self.index
                    return df
                for col in arg:
                    df[col] = self[col]
                df.index = self.index
            return df
        elif isinstance(arg, DataFrame):
            return self.mask(arg)
        else:
            msg = "__getitem__ on type {!r} is not supported"
            raise TypeError(msg.format(type(arg)))

    def mask(self, other):
        df = self.copy()
        for col in self.columns:
            if col in other.columns:
                if other[col].has_nulls:
                    raise ValueError("Column must have no nulls.")

                boolbits = cudautils.compact_mask_bytes(
                    other[col]._column.data_array_view
                )
            else:
                boolbits = cudautils.make_empty_mask(len(self[col]))
            df[col] = df[col].set_mask(boolbits)
        return df

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
                        value = utils.scalar_broadcast_to(value, len(self))
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
                    if is_scalar(value):
                        value = utils.scalar_broadcast_to(value, len(self))
                    self._data[arg] = column.as_column(value)
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

            if is_scalar(value):
                value = column.as_column(
                    utils.scalar_broadcast_to(value, len(self))
                )

            if mask.dtype == "bool":
                if isinstance(value, DataFrame):
                    _setitem_with_dataframe(
                        input_df=self,
                        replace_df=value,
                        input_cols=None,
                        mask=mask,
                    )
                else:
                    for col_name in self._data:
                        self._data[col_name][mask] = column.as_column(value)[
                            mask
                        ]
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
                        self._data[col] = column.as_column(value)
        else:
            msg = "__setitem__ on type {!r} is not supported"
            raise TypeError(msg.format(type(arg)))

    def __delitem__(self, name):
        """
        Drop the given column by *name*.
        """
        self._drop_column(name)

    def __sizeof__(self):
        columns = sum(col.__sizeof__() for col in self._data.values())
        index = self._index.__sizeof__()
        return columns + index

    def memory_usage(self, index=True, deep=False):
        ind = list(self.columns)
        sizes = [col._memory_usage(deep=deep) for col in self._data.values()]
        if index:
            ind.append("Index")
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
    def empty(self):
        return not len(self)

    @property
    def values(self):

        return cupy.asarray(self.as_gpu_matrix())

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

    def astype(self, dtype, errors="raise", **kwargs):
        return self._apply_support_method(
            "astype", dtype=dtype, errors=errors, **kwargs
        )

    def get_renderable_dataframe(self):
        nrows = np.max([pd.options.display.max_rows, 1])
        if pd.options.display.max_rows == 0:
            nrows = len(self)
        ncols = (
            pd.options.display.max_columns
            if pd.options.display.max_columns
            else pd.options.display.width / 2
        )
        if len(self) <= nrows and len(self.columns) <= ncols:
            output = self.copy(deep=False)
        else:
            left_cols = len(self.columns)
            right_cols = 0
            upper_rows = len(self)
            lower_rows = 0
            if len(self) > nrows and nrows > 0:
                upper_rows = int(nrows / 2.0) + 1
                lower_rows = upper_rows + (nrows % 2)
            if len(self.columns) > ncols:
                right_cols = len(self.columns) - int(ncols / 2.0) - 1
                left_cols = int(ncols / 2.0) + 1
            upper_left = self.head(upper_rows).iloc[:, :left_cols]
            upper_right = self.head(upper_rows).iloc[:, right_cols:]
            lower_left = self.tail(lower_rows).iloc[:, :left_cols]
            lower_right = self.tail(lower_rows).iloc[:, right_cols:]
            upper = cudf.concat([upper_left, upper_right], axis=1)
            lower = cudf.concat([lower_left, lower_right], axis=1)
            output = cudf.concat([upper, lower])
        temp_mi_columns = output.columns
        for col in output._data:
            if (
                self._data[col].has_nulls
                and not self._data[col].dtype == "O"
                and not is_datetime_dtype(self._data[col].dtype)
            ):
                output[col] = (
                    output._data[col].astype("str").str().fillna("null")
                )
            else:
                output[col] = output._data[col]
        if isinstance(self.columns, cudf.MultiIndex):
            output.columns = temp_mi_columns
        return output

    def __repr__(self):
        output = self.get_renderable_dataframe()
        # the below is permissible: null in a datetime to_pandas() becomes
        # NaT, which is then replaced with null in this processing step.
        # It is not possible to have a mix of nulls and NaTs in datetime
        # columns because we do not support NaT - pyarrow as_column
        # preprocessing converts NaT input values from numpy or pandas into
        # null.
        output = output.to_pandas().__repr__().replace(" NaT", "null")
        lines = output.split("\n")
        if lines[-1].startswith("["):
            lines = lines[:-1]
            lines.append(
                "[%d rows x %d columns]" % (len(self), len(self.columns))
            )
        return "\n".join(lines)

    def _repr_html_(self):
        lines = (
            self.get_renderable_dataframe()
            .to_pandas()
            ._repr_html_()
            .split("\n")
        )
        if lines[-2].startswith("<p>"):
            lines = lines[:-2]
            lines.append(
                "<p>%d rows × %d columns</p>" % (len(self), len(self.columns))
            )
            lines.append("</div>")
        return "\n".join(lines)

    def _repr_latex_(self):
        return self.get_renderable_dataframe().to_pandas()._repr_latex_()

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

            lhs, rhs = _align_indices(self, other)
            result.index = lhs.index
            max_num_rows = max(lhs.shape[0], rhs.shape[0])

            def fallback(col, fn):
                if fill_value is None:
                    return Series.from_masked_array(
                        data=rmm.device_array(max_num_rows, dtype="float64"),
                        mask=cudautils.make_empty_mask(max_num_rows),
                    ).set_index(col.index)
                else:
                    return getattr(col, fn)(fill_value)

            for col in lhs._data:
                if col not in rhs._data:
                    result[col] = fallback(lhs[col], fn)
            for col in rhs._data:
                if col in lhs._data:
                    result[col] = op(lhs[col], rhs[col])
                else:
                    result[col] = fallback(rhs[col], _reverse_op(fn))
        elif isinstance(other, Series):
            raise NotImplementedError(
                "Series to DataFrame arithmetic not supported "
                "until strings can be used as indices. Try converting your"
                " Series into a DataFrame first."
            )
        elif isinstance(other, numbers.Number):
            for col in self._data:
                result[col] = op(self[col], other)
        else:
            raise NotImplementedError(
                "DataFrame operations with " + str(type(other)) + " not "
                "supported at this time."
            )
        return result

    def add(self, other, fill_value=None):
        return self._apply_op("add", other, fill_value)

    def __add__(self, other):
        return self._apply_op("__add__", other)

    def radd(self, other, fill_value=None):
        return self._apply_op("radd", other, fill_value)

    def __radd__(self, other):
        return self._apply_op("__radd__", other)

    def sub(self, other, fill_value=None):
        return self._apply_op("sub", other, fill_value)

    def __sub__(self, other):
        return self._apply_op("__sub__", other)

    def rsub(self, other, fill_value=None):
        return self._apply_op("rsub", other, fill_value)

    def __rsub__(self, other):
        return self._apply_op("__rsub__", other)

    def mul(self, other, fill_value=None):
        return self._apply_op("mul", other, fill_value)

    def __mul__(self, other):
        return self._apply_op("__mul__", other)

    def rmul(self, other, fill_value=None):
        return self._apply_op("rmul", other, fill_value)

    def __rmul__(self, other):
        return self._apply_op("__rmul__", other)

    def mod(self, other, fill_value=None):
        return self._apply_op("mod", other, fill_value)

    def __mod__(self, other):
        return self._apply_op("__mod__", other)

    def rmod(self, other, fill_value=None):
        return self._apply_op("rmod", other, fill_value)

    def __rmod__(self, other):
        return self._apply_op("__rmod__", other)

    def pow(self, other, fill_value=None):
        return self._apply_op("pow", other, fill_value)

    def __pow__(self, other):
        return self._apply_op("__pow__", other)

    def rpow(self, other, fill_value=None):
        return self._apply_op("rpow", other, fill_value)

    def __rpow__(self, other):
        return self._apply_op("__pow__", other)

    def floordiv(self, other, fill_value=None):
        return self._apply_op("floordiv", other, fill_value)

    def __floordiv__(self, other):
        return self._apply_op("__floordiv__", other)

    def rfloordiv(self, other, fill_value=None):
        return self._apply_op("rfloordiv", other, fill_value)

    def __rfloordiv__(self, other):
        return self._apply_op("__rfloordiv__", other)

    def truediv(self, other, fill_value=None):
        return self._apply_op("truediv", other, fill_value)

    def __truediv__(self, other):
        return self._apply_op("__truediv__", other)

    def rtruediv(self, other, fill_value=None):
        return self._apply_op("rtruediv", other, fill_value)

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

    def equals(self, other):
        for col in self.columns:
            if col not in other.columns:
                return False
            if not self[col].equals(other[col]):
                return False
        if not self.index.equals(other.index):
            return False
        return True

    def iteritems(self):
        """ Iterate over column names and series pairs """
        for k in self:
            yield (k, self[k])

    @property
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
        """
        return _DataFrameLocIndexer(self)

    @property
    def iloc(self):
        """
        Selecting rows and column by position.

        Examples
        --------
        >>> df = DataFrame([('a', list(range(20))),
        ...                 ('b', list(range(20))),
        ...                 ('c', list(range(20)))])

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
        """
        return _DataFrameIlocIndexer(self)

    def iat(self):
        """
        Alias for ``DataFrame.iloc``; provided for compatibility with Pandas.
        """
        return self.iloc

    def at(self):
        """
        Alias for ``DataFrame.loc``; provided for compatibility with Pandas.
        """
        return self.loc

    @property
    def columns(self):
        """Returns a tuple of columns
        """
        if hasattr(self, "multi_cols"):
            return self.multi_cols
        else:
            name = self._columns_name
            return pd.Index(self._data.keys(), name=name)

    @columns.setter
    def columns(self, columns):
        if isinstance(columns, cudf.MultiIndex):
            if len(columns) != len(self.columns):
                msg = (
                    f"Length mismatch: Expected axis has %d elements, "
                    "new values have %d elements"
                    % (len(self.columns), len(columns))
                )
                raise ValueError(msg)
            """
            new_names = []
            for idx, name in enumerate(columns):
                new_names.append(name)
            self._rename_columns(new_names)
            """
            self.multi_cols = columns
        elif isinstance(columns, pd.MultiIndex):
            self.columns = cudf.MultiIndex.from_pandas(columns)
        else:
            if hasattr(self, "multi_cols"):
                delattr(self, "multi_cols")
            self._rename_columns(columns)
            if hasattr(columns, "name"):
                self._columns_name = columns.name

    @property
    def _columns(self):
        """
        Return a list of Column objects backing this dataframe
        """
        return list(self._data.values())

    def _rename_columns(self, new_names):
        old_cols = iter(self._data.keys())
        l_old_cols = len(self._data)
        l_new_cols = len(new_names)
        if l_new_cols != l_old_cols:
            msg = (
                f"Length of new column names: {l_new_cols} does not "
                "match length of previous column names: {l_old_cols}"
            )
            raise ValueError(msg)

        mapper = dict(zip(old_cols, new_names))
        self.rename(mapper=mapper, inplace=True)

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
                    f"Length mismatch: Expected axis has "
                    "%d elements, new values "
                    "have %d elements" % (len(self), len(value))
                )
                raise ValueError(msg)
            self._index = value
            return

        new_length = len(value)
        old_length = len(self._index)

        if len(self._data) > 0 and new_length != old_length:
            msg = (
                f"Length mismatch: Expected axis has "
                "%d elements, new values "
                "have %d elements" % (old_length, new_length)
            )
            raise ValueError(msg)

        # try to build an index from generic _index
        idx = as_index(value)
        self._index = idx

    def reindex(
        self, labels=None, axis=0, index=None, columns=None, copy=True
    ):
        """Return a new DataFrame whose axes conform to a new index

        ``DataFrame.reindex`` supports two calling conventions
        * ``(index=index_labels, columns=column_names)``
        * ``(labels, axis={0 or 'index', 1 or 'columns'})``

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
                                columns=['key', 'val', 'sum'])
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
            idx = idx if isinstance(idx, Index) else as_index(idx)
            if df.index.dtype != idx.dtype:
                cols = cols if cols is not None else list(df.columns)
                df = DataFrame()
            else:
                df = DataFrame(None, idx).join(df, how="left", sort=True)
                # double-argsort to map back from sorted to unsorted positions
                df = df.take(idx.argsort(True).argsort(True))

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

    def set_index(self, index, drop=True):
        """Return a new DataFrame with a new index

        Parameters
        ----------
        index : Index, Series-convertible, str, or list of str
            Index : the new index.
            Series-convertible : values for the new index.
            str : name of column to be used as series
            list of str : name of columns to be converted to a MultiIndex
        drop : boolean
            whether to drop corresponding column for str index argument
        """
        # When index is a list of column names
        if isinstance(index, list):
            if len(index) > 1:
                df = self.copy(deep=False)
                if drop:
                    df = df.drop(columns=index)
                return df.set_index(
                    cudf.MultiIndex.from_frame(self[index], names=index)
                )
            index = index[0]  # List contains single item

        # When index is a column name
        if isinstance(index, str):
            df = self.copy(deep=False)
            if drop:
                df._drop_column(index)
            return df.set_index(self[index])
        # Otherwise
        else:
            index = index if isinstance(index, Index) else as_index(index)
            df = self.copy(deep=False)
            df.index = index
            return df

    def reset_index(self, drop=False, inplace=False):
        out = DataFrame()
        if not drop:
            if isinstance(self.index, cudf.core.multiindex.MultiIndex):
                framed = self.index.to_frame()
                for c in framed.columns:
                    out[c] = framed[c]
            else:
                name = "index"
                if self.index.name is not None:
                    name = self.index.name
                out[name] = self.index
            for c in self.columns:
                out[c] = self[c]
        else:
            out = self
        if inplace is True:
            for column_name in set(out.columns) - set(self.columns):
                self[column_name] = out[column_name]
                self._data.move_to_end(column_name, last=False)
            self.index = RangeIndex(len(self))
        else:
            return out.set_index(RangeIndex(len(self)))

    def take(self, positions, ignore_index=False):
        if ignore_index:
            index = RangeIndex(len(positions))
        else:
            index = self.index.take(positions)
        out = DataFrame()
        if self._data:
            for i, col_name in enumerate(self._data.keys()):
                out[col_name] = self._data[col_name][positions]
        return out.set_index(index)

    def _take_columns(self, positions):
        positions = Series(positions)
        column_names = list(self._data.keys())
        column_values = list(self._data.values())
        result = DataFrame()
        for idx in range(len(positions)):
            if len(self) == 0:
                result[idx] = as_column([])
            else:
                result[column_names[positions[idx]]] = column_values[
                    positions[idx]
                ]
        result.index = self._index
        return result

    def copy(self, deep=True):
        """
        Returns a copy of this dataframe

        Parameters
        ----------
        deep: bool
           Make a full copy of Series columns and Index at the GPU level, or
           create a new allocation with references.
        """
        data = OrderedDict()

        if deep:
            index = self._index.copy(deep)
            for k in self._data:
                data[k] = self._data[k].copy(deep)
        else:
            index = self._index
            for k in self._data:
                data[k] = self._data[k]

        out = DataFrame(data=data, columns=self.columns.copy(deep=deep))

        out.index = index

        return out

    def __copy__(self):
        return self.copy(deep=True)

    def __deepcopy__(self, memo={}):
        """
        Parameters
        ----------
        memo, default None
            Standard signature. Unused
        """
        if memo is None:
            memo = {}
        return self.copy(deep=True)

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
            index = None
            if isinstance(value, (pd.Series, Series)):
                index = as_index(value.index)
            len_value = len(value)
            if len_value > 0:
                if index is None:
                    index = RangeIndex(start=0, stop=len_value)
                if num_cols != 0:
                    for col_name in self._data:
                        self._data[col_name] = column.column_empty_like(
                            self._data[col_name],
                            masked=True,
                            newsize=len_value,
                        )
            if index is None:
                index = RangeIndex(start=0, stop=0)
            self._index = index

        value = column.as_column(value)

        self._data[name] = value
        keys = list(self._data.keys())
        for i, col in enumerate(keys[loc:-1]):
            self._data.move_to_end(col)

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

        self.insert(len(self.columns), name, data)

    def drop(
        self,
        labels=None,
        axis=None,
        columns=None,
        errors="raise",
        inplace=False,
    ):
        """Drop column(s)

        Parameters
        ----------
        labels : str or sequence of strings
            Name of column(s) to be dropped.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Only axis=1 is currently supported.
        columns: array of column names, the same as using labels and axis=1
        errors : {'ignore', 'raise'}, default 'raise'
            This parameter is currently ignored.
        inplace : bool, default False
            If True, do operation inplace and return `self`.

        Returns
        -------
        A dataframe without dropped column(s)

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df['key'] = [0, 1, 2, 3, 4]
        >>> df['val'] = [float(i + 10) for i in range(5)]
        >>> df_new = df.drop('val')
        >>> print(df)
           key   val
        0    0  10.0
        1    1  11.0
        2    2  12.0
        3    3  13.0
        4    4  14.0
        >>> print(df_new)
           key
        0    0
        1    1
        2    2
        3    3
        4    4
        """
        if axis == 0 and labels is not None:
            raise NotImplementedError("Can only drop columns, not rows")
        if errors != "raise":
            raise NotImplementedError("errors= keyword not implemented")
        if labels is None and columns is None:
            raise ValueError(
                "Need to specify at least one of 'labels' or 'columns'"
            )
        if labels is not None and columns is not None:
            raise ValueError("Cannot specify both 'labels' and 'columns'")

        if labels is not None:
            target = labels
        else:
            target = columns

        columns = (
            [target]
            if isinstance(target, (str, numbers.Number))
            else list(target)
        )
        if inplace:
            outdf = self
        else:
            outdf = self.copy()
        for c in columns:
            outdf._drop_column(c)
        return outdf

    def drop_column(self, name):
        """Drop a column by *name*
        """
        warnings.warn(
            "The drop_column method is deprecated. "
            "Use the drop method instead.",
            DeprecationWarning,
        )
        self._drop_column(name)

    def _drop_column(self, name):
        """Drop a column by *name*
        """
        if name not in self._data:
            raise NameError("column {!r} does not exist".format(name))
        del self._data[name]

    def drop_duplicates(self, subset=None, keep="first", inplace=False):
        """
        Return DataFrame with duplicate rows removed, optionally only
        considering certain subset of columns.
        """
        in_cols = list(self._data.values())
        if subset is None:
            subset = self._data
        elif (
            not np.iterable(subset)
            or isinstance(subset, str)
            or isinstance(subset, tuple)
            and subset in self.columns
        ):
            subset = (subset,)
        diff = set(subset) - set(self._data)
        if len(diff) != 0:
            raise KeyError("columns {!r} do not exist".format(diff))
        subset_cols = [
            col for name, col in self._data.items() if name in subset
        ]
        in_index = self.index
        if isinstance(in_index, cudf.core.multiindex.MultiIndex):
            in_index = RangeIndex(len(in_index), name=in_index.name)
        out_cols, new_index = libcudf.stream_compaction.drop_duplicates(
            [in_index.as_column()], in_cols, subset_cols, keep
        )
        new_index = as_index(new_index, name=self.index.name)
        if isinstance(self.index, cudf.core.multiindex.MultiIndex):
            new_index = self.index.take(new_index)

        outdf = DataFrame()
        for k, new_col in zip(self._data, out_cols):
            outdf[k] = new_col
        outdf = outdf.set_index(new_index)

        return self._mimic_inplace(outdf, inplace=inplace)

    def _mimic_inplace(self, result, inplace=False):
        if inplace:
            self._data = result._data
            self._index = result._index
            if hasattr(result, "multi_cols"):
                self.multi_cols = result.multi_cols
            self._columns_name = result._columns_name
        else:
            return result

    def dropna(self, axis=0, how="any", subset=None, thresh=None):
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
        subset : list, optional
            List of columns to consider when dropping rows (all columns
            are considered by default). Alternatively, when dropping
            columns, subset is a list of rows to consider.
        thresh: int, optional
            If specified, then drops every row (or column) containing
            less than `thresh` non-null values


        Returns
        -------
        Copy of the DataFrame with rows/columns containing nulls dropped.
        """
        if axis == 0:
            return self._drop_na_rows(how=how, subset=subset, thresh=thresh)
        else:
            return self._drop_na_columns(how=how, subset=subset, thresh=thresh)

    def _drop_na_rows(self, how="any", subset=None, thresh=None):
        """
        Drop rows containing nulls.
        """
        data_cols = self._columns

        index_cols = []
        if isinstance(self.index, cudf.MultiIndex):
            index_cols.extend(self.index._source_data._columns)
        else:
            index_cols.append(self.index.as_column())

        input_cols = index_cols + data_cols

        if subset is not None:
            subset = self._columns_view(subset)._columns
        else:
            subset = self._columns

        if len(subset) == 0:
            return self

        result_cols = libcudf.stream_compaction.drop_nulls(
            input_cols, how=how, subset=subset, thresh=thresh
        )

        result_index_cols, result_data_cols = (
            result_cols[: len(index_cols)],
            result_cols[len(index_cols) :],
        )

        if isinstance(self.index, cudf.MultiIndex):
            result_index = cudf.MultiIndex.from_frame(
                DataFrame._from_columns(result_index_cols),
                names=self.index.names,
            )
        else:
            result_index = cudf.core.index.as_index(result_index_cols[0])

        df = DataFrame._from_columns(
            result_data_cols, index=result_index, columns=self.columns
        )
        return df

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

        for col in self.columns:
            if (len(df[col]) - df[col].null_count) < thresh:
                continue
            out_cols.append(col)

        return self[out_cols]

    def pop(self, item):
        """Return a column and drop it from the DataFrame.
        """
        popped = self[item]
        del self[item]
        return popped

    def rename(self, mapper=None, columns=None, copy=True, inplace=False):
        """
        Alter column labels.

        Function / dict values must be unique (1-to-1). Labels not contained in
        a dict / Series will be left as-is. Extra labels listed don’t throw an
        error.

        Parameters
        ----------
        mapper, columns : dict-like or function, optional
            dict-like or functions transformations to apply to
            the column axis' values.
        copy : boolean, default True
            Also copy underlying data
        inplace: boolean, default False
            Return new DataFrame.  If True, assign columns without copy

        Returns
        -------
        DataFrame

        Notes
        -----
        Difference from pandas:
          * Support axis='columns' only.
          * Not supporting: index, level

        Rename will not overwite column names. If a list with duplicates it
        passed, column names will be postfixed.
        """
        # Pandas defaults to using columns over mapper
        if columns:
            mapper = columns

        out = DataFrame(index=self.index)
        if isinstance(mapper, Mapping):
            postfix = 1
            # It is possible for DataFrames with a MultiIndex columns object
            # to have columns with the same name. The followig use of
            # _cols.items and ("cudf_"... allows the use of rename in this case
            for key, col in self._data.items():
                if key in mapper:
                    if mapper[key] in out.columns:
                        out_column = mapper[key] + ("cudf_" + str(postfix),)
                        postfix += 1
                    else:
                        out_column = mapper[key]
                    out[out_column] = col
                else:
                    out[key] = col
        elif callable(mapper):
            for col in self.columns:
                out[mapper(col)] = self[col]

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

    @classmethod
    def _concat(cls, objs, axis=0, ignore_index=False):

        libcudf.nvtx.nvtx_range_push("CUDF_CONCAT", "orange")

        if ignore_index:
            index = RangeIndex(sum(map(len, objs)))
        elif isinstance(objs[0].index, cudf.core.multiindex.MultiIndex):
            index = cudf.core.multiindex.MultiIndex._concat(
                [o.index for o in objs]
            )
        else:
            index = Index._concat([o.index for o in objs])

        # Currently we only support sort = False
        # Change below when we want to support sort = True
        # below functions as an ordered set
        all_columns_ls = [col for o in objs for col in o.columns]
        unique_columns_ordered_ls = OrderedDict.fromkeys(all_columns_ls).keys()

        # Concatenate cudf.series for all columns

        data = {
            i: Series._concat(
                [
                    o[c]
                    if c in o.columns
                    else utils.get_null_series(size=len(o), dtype=np.bool)
                    for o in objs
                ],
                index=index,
            )
            for i, c in enumerate(unique_columns_ordered_ls)
        }

        out = cls(data)

        out.index = index

        if isinstance(objs[0].columns, cudf.MultiIndex):
            out.columns = objs[0].columns
        else:
            out.columns = unique_columns_ordered_ls

        libcudf.nvtx.nvtx_range_pop()
        return out

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
        A (nrow x ncol) numpy ndarray in "F" order.
        """
        if columns is None:
            columns = self.columns

        cols = [self._data[k] for k in columns]
        ncol = len(cols)
        nrow = len(self)
        if ncol < 1:
            raise ValueError("require at least 1 column")
        if nrow < 1:
            raise ValueError("require at least 1 row")
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

        if order == "F":
            matrix = rmm.device_array(
                shape=(nrow, ncol), dtype=dtype, order=order
            )
            for colidx, inpcol in enumerate(cols):
                dense = inpcol.astype(dtype).to_gpu_array(fillna="pandas")
                matrix[:, colidx].copy_to_device(dense)
        elif order == "C":
            matrix = cudautils.row_matrix(cols, nrow, ncol, dtype)
        else:
            errmsg = (
                "order parameter should be 'C' for row major or 'F' for"
                "column major GPU matrix"
            )
            raise ValueError(errmsg.format(k))
        return matrix

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
        if hasattr(cats, "to_pandas"):
            cats = cats.to_pandas()
        else:
            cats = pd.Series(cats)

        newnames = [prefix_sep.join([prefix, str(cat)]) for cat in cats]
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

    def argsort(self, ascending=True, na_position="last"):
        cols = list(self._data.values())
        return get_sorted_inds(
            cols, ascending=ascending, na_position=na_position
        )

    def sort_index(self, ascending=True):
        """Sort by the index
        """
        return self.take(self.index.argsort(ascending=ascending))

    def sort_values(self, by, ascending=True, na_position="last"):
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
        # argsort the `by` column
        return self.take(
            self[by].argsort(ascending=ascending, na_position=na_position)
        )

    def nlargest(self, n, columns, keep="first"):
        """Get the rows of the DataFrame sorted by the n largest value of *columns*

        Notes
        -----
        Difference from pandas:
        * Only a single column is supported in *columns*
        """
        return self._n_largest_or_smallest("nlargest", n, columns, keep)

    def nsmallest(self, n, columns, keep="first"):
        """Get the rows of the DataFrame sorted by the n smallest value of *columns*

        Difference from pandas:
        * Only a single column is supported in *columns*
        """
        return self._n_largest_or_smallest("nsmallest", n, columns, keep)

    def _n_largest_or_smallest(self, method, n, columns, keep):
        # Get column to operate on
        if not isinstance(columns, str):
            [column] = columns
        else:
            column = columns
        if not (0 <= n <= len(self)):
            raise ValueError("n out-of-bound")
        col = self[column].reset_index(drop=True)
        # Operate
        sorted_series = getattr(col, method)(n=n, keep=keep)
        df = DataFrame()
        new_positions = sorted_series.index.gpu_values
        for k in self.columns:
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
        Not supporting *copy* because default and only behaviour is copy=True
        """
        # Never transpose a MultiIndex - remove the existing columns and
        # replace with a RangeIndex. Afterward, reassign.
        inp = self.copy(deep=False)
        temp_columns = inp.columns.copy(deep=False)
        temp_index = inp.index.copy(deep=False)
        if len(self._data) == 0:
            return DataFrame(index=temp_columns, columns=temp_index)
        inp.columns = pd.RangeIndex(start=0, stop=len(self.columns))
        inp.index = RangeIndex(start=0, stop=len(self))
        result = libcudf.transpose.transpose(inp)
        result._index = as_index(temp_columns)
        if not isinstance(temp_index, cudf.MultiIndex):
            temp_index = temp_index.to_pandas()
        result.columns = temp_index
        return result

    @property
    def T(self):
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

    def merge(
        self,
        right,
        on=None,
        how="inner",
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
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
              outer join; preserve key order.
            - right : not supported.
            - outer : use union of keys from both frames, similar to a SQL
              full outer join; sort keys lexicographically.
            - inner: use intersection of keys from both frames, similar to
              a SQL inner join; preserve the order of the left keys.
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
            Sort the join keys lexicographically in the result DataFrame.
            If False, the order of the join keys depends on the join type
            (see the `how` keyword).
        suffixes: Tuple[str, str], defaults to ('_x', '_y')
            Suffixes applied to overlapping column names on the left and right
            sides
        method : {‘hash’, ‘sort’}, default ‘hash’
            The implementation method to be used for the operation.

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
        """
        libcudf.nvtx.nvtx_range_push("CUDF_JOIN", "blue")
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
        if how not in ["left", "inner", "outer"]:
            raise NotImplementedError(
                "{!r} merge not supported yet".format(how)
            )

        # Making sure that the "on" arguments are list of column names
        if on:
            on = [on] if isinstance(on, str) else list(on)
        if left_on:
            left_on = [left_on] if isinstance(left_on, str) else list(left_on)
        if right_on:
            right_on = (
                [right_on] if isinstance(right_on, str) else list(right_on)
            )

        lhs = self.copy(deep=False)
        rhs = right.copy(deep=False)

        same_named_columns = set(lhs.columns) & set(rhs.columns)

        # Since GDF doesn't take indexes, we insert indexes as regular columns.
        # In order to do that we need some unique column names
        result_index_name = _unique_name(
            itertools.chain(lhs.columns, rhs.columns), suffix="_result_index"
        )
        merge_index_name = _unique_name(
            itertools.chain(lhs.columns, rhs.columns), suffix="_merge_index"
        )

        # Let's find the columns to do the merge on.
        if left_index and right_index:
            lhs[merge_index_name] = lhs.index
            rhs[merge_index_name] = rhs.index
            left_on = right_on = [merge_index_name]
        elif on:
            if left_on or right_on:
                raise ValueError(
                    'Can only pass argument "on" OR "left_on" '
                    'and "right_on", not a combination of both.'
                )
            left_on = right_on = on
        elif left_index and right_on:
            if len(right_on) != 1:  # TODO: support multi-index
                raise ValueError("right_on should be a single column")
            lhs[merge_index_name] = lhs.index
            left_on = [merge_index_name]
            rhs[result_index_name] = rhs.index
        elif right_index and left_on:
            if len(left_on) != 1:  # TODO: support multi-index
                raise ValueError("left_on should be a single column")
            rhs[merge_index_name] = rhs.index
            right_on = [merge_index_name]
            lhs[result_index_name] = lhs.index
        elif not (left_on or right_on):
            left_on = right_on = list(same_named_columns)
            if len(left_on) == 0:
                raise ValueError("No common columns to perform merge on")
        else:
            if len(right_on) != len(left_on):
                raise ValueError(
                    "right_on and left_on must have same " "number of columns"
                )

        # Fix column names by appending `suffixes`
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
                else:
                    lhs.rename({name: "%s%s" % (name, lsuffix)}, inplace=True)
                    rhs.rename({name: "%s%s" % (name, rsuffix)}, inplace=True)
                    if name in left_on:
                        left_on[left_on.index(name)] = "%s%s" % (name, lsuffix)
                    if name in right_on:
                        right_on[right_on.index(name)] = "%s%s" % (
                            name,
                            rsuffix,
                        )

        # We save the original categories for the reconstruction of the
        # final data frame
        categorical_dtypes = {}
        for name, col in itertools.chain(lhs._data.items(), rhs._data.items()):
            if is_categorical_dtype(col):
                categorical_dtypes[name] = col.dtype

        # Save the order of the original column names for preservation later
        org_names = list(itertools.chain(lhs._data.keys(), rhs._data.keys()))

        # Compute merge
        gdf_result = libcudf.join.join(
            lhs._data, rhs._data, left_on, right_on, how, method
        )

        # Let's sort the columns of the GDF result. NB: Pandas doc says
        # that it sorts when how='outer' but this is NOT the case.
        result = []
        if sort:
            # Pandas lexicographically sort is NOT a sort of all columns.
            # Instead, it sorts columns in lhs, then in "on", and then rhs.
            left_of_on = []
            for name in lhs._data.keys():
                if name not in left_on:
                    for i in range(len(gdf_result)):
                        if gdf_result[i][1] == name:
                            left_of_on.append(gdf_result.pop(i))
                            break
            in_on = []
            for name in itertools.chain(lhs._data.keys(), rhs._data.keys()):
                if name in left_on or name in right_on:
                    for i in range(len(gdf_result)):
                        if gdf_result[i][1] == name:
                            in_on.append(gdf_result.pop(i))
                            break
            right_of_on = []
            for name in rhs._data.keys():
                if name not in right_on:
                    for i in range(len(gdf_result)):
                        if gdf_result[i][1] == name:
                            right_of_on.append(gdf_result.pop(i))
                            break
            result = (
                sorted(left_of_on, key=lambda x: str(x[1]))
                + sorted(in_on, key=lambda x: str(x[1]))
                + sorted(right_of_on, key=lambda x: str(x[1]))
            )
        else:
            for org_name in org_names:
                for i in range(len(gdf_result)):
                    if gdf_result[i][1] == org_name:
                        result.append(gdf_result.pop(i))
                        break
            assert len(gdf_result) == 0

        # Build a new data frame based on the merged columns from GDF
        df = DataFrame()
        for col, name in result:
            if is_string_dtype(col):
                df[name] = col
            elif is_categorical_dtype(categorical_dtypes.get(name, col.dtype)):
                dtype = categorical_dtypes.get(name, col.dtype)
                df[name] = column.build_categorical_column(
                    categories=dtype.categories,
                    codes=col,
                    mask=col.mask,
                    ordered=dtype.ordered,
                )
            else:
                df[name] = column.build_column(
                    col.data,
                    dtype=categorical_dtypes.get(name, col.dtype),
                    mask=col.mask,
                )

        # Let's make the "index as column" back into an index
        if left_index and right_index:
            df.index = df[merge_index_name]
            df.index.name = lhs.index.name
        elif result_index_name in df.columns:
            df.index = df[result_index_name]
            if left_index:
                df.index.name = rhs.index.name
            elif right_index:
                df.index.name = lhs.index.name

        # Remove all of the "index as column" columns
        if merge_index_name in df.columns:
            df._drop_column(merge_index_name)
        if result_index_name in df.columns:
            df._drop_column(result_index_name)

        libcudf.nvtx.nvtx_range_pop()

        return df

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

        libcudf.nvtx.nvtx_range_push("CUDF_JOIN", "blue")

        # Outer joins still use the old implementation
        if type != "":
            warnings.warn(
                'type="' + type + '" parameter is deprecated.'
                'Use method="' + type + '" instead.',
                DeprecationWarning,
            )
            method = type

        if how not in ["left", "right", "inner", "outer"]:
            raise NotImplementedError("unsupported {!r} join".format(how))

        if how == "right":
            # libgdf doesn't support right join directly, we will swap the
            # dfs and use left join
            return other.join(
                self,
                other,
                how="left",
                lsuffix=rsuffix,
                rsuffix=lsuffix,
                sort=sort,
                method="hash",
            )

        same_names = set(self.columns) & set(other.columns)
        if same_names and not (lsuffix or rsuffix):
            raise ValueError(
                "there are overlapping columns but "
                "lsuffix and rsuffix are not defined"
            )

        lhs = DataFrame()
        rhs = DataFrame()

        idx_col_names = []
        if isinstance(self.index, cudf.core.multiindex.MultiIndex):
            if not isinstance(other.index, cudf.core.multiindex.MultiIndex):
                raise TypeError(
                    "Left index is MultiIndex, but right index is "
                    + type(other.index)
                )

            index_frame_l = self.index.copy().to_frame(index=False)
            index_frame_r = other.index.copy().to_frame(index=False)

            if (index_frame_l.columns != index_frame_r.columns).any():
                raise ValueError(
                    "Left and Right indice-column names must match."
                )

            for name in index_frame_l.columns:
                idx_col_name = str(uuid.uuid4())
                idx_col_names.append(idx_col_name)

                lhs[idx_col_name] = index_frame_l[name].set_index(self.index)
                rhs[idx_col_name] = index_frame_r[name].set_index(other.index)

        else:
            idx_col_names.append(str(uuid.uuid4()))
            lhs[idx_col_names[0]] = Series(self.index.as_column()).set_index(
                self.index
            )
            rhs[idx_col_names[0]] = Series(other.index.as_column()).set_index(
                other.index
            )

        for name in self.columns:
            lhs[name] = self[name]

        for name in other.columns:
            rhs[name] = other[name]

        lhs = lhs.reset_index(drop=True)
        rhs = rhs.reset_index(drop=True)

        cat_join = []
        for name in idx_col_names:
            if is_categorical_dtype(lhs[name]):

                lcats = lhs[name].cat.categories
                rcats = rhs[name].cat.categories

                def _set_categories(col, cats):
                    return col.cat._set_categories(
                        cats, is_unique=True
                    ).fillna(-1)

                if how == "left":
                    cats = lcats
                    rhs[name] = _set_categories(rhs[name], cats)
                elif how == "right":
                    cats = rcats
                    lhs[name] = _set_categories(lhs[name], cats)
                elif how in ["inner", "outer"]:
                    cats = column.as_column(lcats).append(rcats)
                    cats = Series(cats).drop_duplicates()._column

                    lhs[name] = _set_categories(lhs[name], cats)
                    lhs[name] = lhs[name]._column.as_numerical

                    rhs[name] = _set_categories(rhs[name], cats)
                    rhs[name] = rhs[name]._column.as_numerical

                cat_join.append((name, cats))

        if lsuffix == "":
            lsuffix = "l"
        if rsuffix == "":
            rsuffix = "r"

        df = lhs.merge(
            rhs,
            on=idx_col_names,
            how=how,
            suffixes=(lsuffix, rsuffix),
            method=method,
        )

        for name, cats in cat_join:

            if is_categorical_dtype(df[name]):
                codes = df[name]._column.codes
            else:
                codes = df[name]._column
            df[name] = column.build_categorical_column(
                categories=cats, codes=codes, ordered=False
            )

        if sort and len(df):
            df = df.sort_values(idx_col_names)

        df = df.set_index(idx_col_names)
        # change index to None to better reflect pandas behavior
        df.index.name = None

        if len(idx_col_names) > 1:
            df.index._source_data.columns = index_frame_l.columns
            df.index.names = index_frame_l.columns

        return df

    def groupby(
        self,
        by=None,
        sort=True,
        as_index=True,
        method="hash",
        level=None,
        group_keys=True,
        dropna=True,
    ):
        """Groupby

        Parameters
        ----------
        by : list-of-str or str
            Column name(s) to form that groups by.
        sort : bool, default True
            Force sorting group keys.
        as_index : bool, default True
            Indicates whether the grouped by columns become the index
            of the returned DataFrame
        method : str, optional
            A string indicating the method to use to perform the group by.
            Valid values are "hash" or "cudf".
            "cudf" method may be deprecated in the future, but is currently
            the only method supporting group UDFs via the `apply` function.
        dropna : bool, optional
            If True (default), drop null keys.
            If False, perform grouping by keys containing null(s).

        Returns
        -------
        The groupby object

        Notes
        -----
        No empty rows are returned.  (For categorical keys, pandas returns
        rows for all categories even if they are no corresponding values.)
        """
        if group_keys is not True:
            raise NotImplementedError(
                "The group_keys keyword is not yet implemented"
            )
        if by is None and level is None:
            raise TypeError(
                "groupby() requires either by or level to be" "specified."
            )
        if method == "cudf":
            from cudf.core.groupby.legacy_groupby import Groupby

            if as_index:
                warnings.warn(
                    "as_index==True not supported due to the lack of "
                    "multi-index with legacy groupby function. Use hash "
                    "method for multi-index"
                )
            result = Groupby(self, by=by)
            return result
        else:
            from cudf.core.groupby.groupby import DataFrameGroupBy

            # The corresponding pop() is in
            # DataFrameGroupBy._apply_aggregation()
            libcudf.nvtx.nvtx_range_push("CUDF_GROUPBY", "purple")

            result = DataFrameGroupBy(
                self,
                by=by,
                method=method,
                as_index=as_index,
                sort=sort,
                level=level,
                dropna=dropna,
            )
            return result

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

    def query(self, expr, local_dict={}):
        """
        Query with a boolean expression using Numba to compile a GPU kernel.

        See pandas.DataFrame.query.

        Parameters
        ----------

        expr : str
            A boolean expression. Names in expression refer to columns.

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
        if self.empty:
            return self.copy()

        if not isinstance(local_dict, dict):
            raise TypeError(
                "local_dict type: expected dict but found {!r}".format(
                    type(local_dict)
                )
            )

        libcudf.nvtx.nvtx_range_push("CUDF_QUERY", "purple")
        # Get calling environment
        callframe = inspect.currentframe().f_back
        callenv = {
            "locals": callframe.f_locals,
            "globals": callframe.f_globals,
            "local_dict": local_dict,
        }
        # Run query
        boolmask = queryutils.query_execute(self, expr, callenv)

        selected = Series(boolmask)
        newdf = DataFrame()
        for col in self.columns:
            newseries = self[col][selected]
            newdf[col] = newseries
        result = newdf
        libcudf.nvtx.nvtx_range_pop()
        return result

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
        kwargs={},
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
        function can be used with any *tpb* in a efficient manner.

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
        """Hash the given *columns* and return a new Series

        Parameters
        ----------
        column : sequence of str; optional
            Sequence of column names. If columns is *None* (unspecified),
            all columns in the frame are used.
        """
        from cudf.core.column import numerical

        if columns is None:
            columns = self.columns

        cols = [self[k]._column for k in columns]
        return Series(numerical.column_hash_values(*cols))

    def partition_by_hash(self, columns, nparts):
        """Partition the dataframe by the hashed value of data in *columns*.

        Parameters
        ----------
        columns : sequence of str
            The names of the columns to be hashed.
            Must have at least one name.
        nparts : int
            Number of output partitions

        Returns
        -------
        partitioned: list of DataFrame
        """
        cols = list(self._data.values())
        names = list(self._data.keys())
        key_indices = [names.index(k) for k in columns]
        # Allocate output buffers
        outputs = [col.copy() for col in cols]
        # Call hash_partition
        offsets = libcudf.hash.hash_partition(
            cols, key_indices, nparts, outputs
        )
        # Re-construct output partitions
        outdf = DataFrame()
        for k, col in zip(self._data, outputs):
            outdf[k] = col
        # Slice into partition
        return [outdf[s:e] for s, e in zip(offsets, offsets[1:] + [None])]

    def replace(self, to_replace, replacement):
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
        replacement : numeric, str, list-like, or dict
            Value(s) to replace `to_replace` with. If a dict is provided, then
            its keys must match the keys in *to_replace*, and correponding
            values must be compatible (e.g., if they are lists, then they must
            match in length).

        Returns
        -------
        result : DataFrame
            DataFrame after replacement.
        """
        outdf = self.copy()

        if not is_dict_like(to_replace):
            to_replace = dict.fromkeys(self.columns, to_replace)
        if not is_dict_like(replacement):
            replacement = dict.fromkeys(self.columns, replacement)

        for k in to_replace:
            outdf[k] = self[k].replace(to_replace[k], replacement[k])

        return outdf

    def fillna(self, value, method=None, axis=None, inplace=False, limit=None):
        """Fill null values with ``value``.

        Parameters
        ----------
        value : scalar, Series-like or dict
            Value to use to fill nulls. If Series-like, null values
            are filled with values in corresponding indices.
            A dict can be used to provide different values to fill nulls
            in different columns.

        Returns
        -------
        result : DataFrame
            Copy with nulls filled.

        Examples
        --------
        >>> import cudf
        >>> gdf = cudf.DataFrame({'a': [1, 2, None], 'b': [3, None, 5]})
        >>> gdf.fillna(4).to_pandas()
        a  b
        0  1  3
        1  2  4
        2  4  5
        >>> gdf.fillna({'a': 3, 'b': 4}).to_pandas()
        a  b
        0  1  3
        1  2  4
        2  3  5
        """
        if inplace:
            outdf = {}  # this dict will just hold Nones
        else:
            outdf = self.copy()

        if not is_dict_like(value):
            value = dict.fromkeys(self.columns, value)

        for k in value:
            outdf[k] = self[k].fillna(
                value[k],
                method=method,
                axis=axis,
                inplace=inplace,
                limit=limit,
            )

        if not inplace:
            return outdf

    def describe(self, percentiles=None, include=None, exclude=None):
        """Compute summary statistics of a DataFrame's columns. For numeric
        data, the output includes the minimum, maximum, mean, median,
        standard deviation, and various quantiles. For object data, the output
        includes the count, number of unique values, the most common value, and
        the number of occurrences of the most common value.

        Parameters
        ----------
        percentiles : list-like, optional
            The percentiles used to generate the output summary statistics.
            If None, the default percentiles used are the 25th, 50th and 75th.
            Values should be within the interval [0, 1].

        include: str, list-like, optional
            The dtypes to be included in the output summary statistics. Columns
            of dtypes not included in this list will not be part of the output.
            If include='all', all dtypes are included. Default of None includes
            all numeric columns.

        exclude: str, list-like, optional
            The dtypes to be excluded from the output summary statistics.
            Columns of dtypes included in this list will not be part of the
            output. Default of None excludes no columns.

        Returns
        -------
        output_frame : DataFrame
            Summary statistics of relevant columns in the original dataframe.

        Examples
        --------
        Describing a ``Series`` containing numeric values.

        >>> import cudf
        >>> s = cudf.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> print(s.describe())
           stats   values
        0  count     10.0
        1   mean      5.5
        2    std  3.02765
        3    min      1.0
        4    25%      2.5
        5    50%      5.5
        6    75%      7.5
        7    max     10.0

        Describing a ``DataFrame``. By default all numeric fields
        are returned.

        >>> gdf = cudf.DataFrame()
        >>> gdf['a'] = [1,2,3]
        >>> gdf['b'] = [1.0, 2.0, 3.0]
        >>> gdf['c'] = ['x', 'y', 'z']
        >>> gdf['d'] = [1.0, 2.0, 3.0]
        >>> gdf['d'] = gdf['d'].astype('float32')
        >>> print(gdf.describe())
           stats    a    b    d
        0  count  3.0  3.0  3.0
        1   mean  2.0  2.0  2.0
        2    std  1.0  1.0  1.0
        3    min  1.0  1.0  1.0
        4    25%  1.5  1.5  1.5
        5    50%  1.5  1.5  1.5
        6    75%  2.5  2.5  2.5
        7    max  3.0  3.0  3.0

        Using the ``include`` keyword to describe only specific dtypes.

        >>> gdf = cudf.DataFrame()
        >>> gdf['a'] = [1,2,3]
        >>> gdf['b'] = [1.0, 2.0, 3.0]
        >>> gdf['c'] = ['x', 'y', 'z']
        >>> print(gdf.describe(include='int'))
           stats    a
        0  count  3.0
        1   mean  2.0
        2    std  1.0
        3    min  1.0
        4    25%  1.5
        5    50%  1.5
        6    75%  2.5
        7    max  3.0
        """

        def _create_output_frame(data, percentiles=None):
            # hack because we don't support strings in indexes
            return DataFrame(
                {
                    col: data[col].describe(percentiles=percentiles)
                    for col in data.columns
                },
                index=Series(column.column_empty(0, dtype="int32"))
                .describe(percentiles=percentiles)
                .index,
            )

        if not include and not exclude:
            numeric_data = self.select_dtypes(np.number)
            output_frame = _create_output_frame(numeric_data, percentiles)

        elif include == "all":
            if exclude:
                raise ValueError("Cannot exclude when include='all'.")

            included_data = self.select_dtypes(np.number)
            output_frame = _create_output_frame(included_data, percentiles)
            logging.warning(
                "Describe does not yet include StringColumns or "
                "DatetimeColumns."
            )

        else:
            if not include:
                include = np.number

            included_data = self.select_dtypes(
                include=include, exclude=exclude
            )
            if included_data.empty:
                raise ValueError("No data of included types.")
            output_frame = _create_output_frame(included_data, percentiles)

        return output_frame

    def isnull(self):
        """Identify missing values in a DataFrame.
        """
        return self._apply_support_method("isnull")

    def isna(self):
        """Identify missing values in a DataFrame. Alias for isnull.
        """
        return self.isnull()

    def notna(self):
        """Identify non-missing values in a DataFrame.
        """
        return self._apply_support_method("notna")

    def notnull(self):
        """Identify non-missing values in a DataFrame. Alias for notna.
        """
        return self.notna()

    def to_pandas(self):
        """
        Convert to a Pandas DataFrame.

        Examples
        --------
        >>> import cudf
        >>> a = ('a', [0, 1, 2])
        >>> b = ('b', [-3, 2, 0])
        >>> df = cudf.DataFrame([a, b])
        >>> type(df.to_pandas())
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
    def from_pandas(cls, dataframe, nan_as_null=True):
        """
        Convert from a Pandas DataFrame.

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
        <cudf.DataFrame ncols=2 nrows=3 >
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("not a pandas.DataFrame")

        df = cls()
        # Set columns
        for i, colk in enumerate(dataframe.columns):
            vals = dataframe[colk].values
            # necessary because multi-index can return multiple
            # columns for a single key
            if len(vals.shape) == 1:
                df[i] = Series(vals, nan_as_null=nan_as_null)
            else:
                vals = vals.T
                if vals.shape[0] == 1:
                    df[i] = Series(vals.flatten(), nan_as_null=nan_as_null)
                else:
                    if isinstance(colk, tuple):
                        colk = str(colk)
                    for idx in range(len(vals.shape)):
                        df[i] = Series(vals[idx], nan_as_null=nan_as_null)

        # Set columns
        if isinstance(dataframe.columns, pd.MultiIndex):
            df.columns = cudf.MultiIndex.from_pandas(dataframe.columns)
        else:
            df.columns = dataframe.columns

        # Set index
        if isinstance(dataframe.index, pd.MultiIndex):
            index = cudf.from_pandas(dataframe.index)
        else:
            index = dataframe.index
        result = df.set_index(index)

        return result

    def to_arrow(self, preserve_index=True):
        """
        Convert to a PyArrow Table.

        Examples
        --------
        >>> import cudf
        >>> a = ('a', [0, 1, 2])
        >>> b = ('b', [-3, 2, 0])
        >>> df = cudf.DataFrame([a, b])
        >>> df.to_arrow()
        pyarrow.Table
        None: int64
        a: int64
        b: int64
        """
        arrays = []
        names = []
        types = []
        index_names = []
        index_columns = []
        index_descriptors = []

        for name, col in self._data.items():
            names.append(name)
            arrow_col = col.to_arrow()
            arrays.append(arrow_col)
            types.append(arrow_col.type)

        index_name = pa.pandas_compat._index_level_name(self.index, 0, names)
        index_columns.append(self.index)

        # It would be better if we didn't convert this if we didn't have to,
        # but we first need better tooling for cudf --> pyarrow type
        # conversions
        if preserve_index:
            if isinstance(self.index, cudf.core.index.RangeIndex):
                descr = {
                    "kind": "range",
                    "name": self.index.name,
                    "start": self.index._start,
                    "stop": self.index._stop,
                    "step": 1,
                }
            else:
                index_arrow = self.index.to_arrow()
                descr = index_name
                types.append(index_arrow.type)
                arrays.append(index_arrow)
                names.append(index_name)
                index_names.append(index_name)
            index_descriptors.append(descr)

        # We may want to add additional metadata to this in the future, but
        # for now lets just piggyback off of what's done for Pandas
        metadata = pa.pandas_compat.construct_metadata(
            self,
            names,
            index_columns,
            index_descriptors,
            preserve_index,
            types,
        )

        return pa.Table.from_arrays(arrays, names=names, metadata=metadata)

    @classmethod
    def from_arrow(cls, table):
        """Convert from a PyArrow Table.

        Raises
        ------
        TypeError for invalid input type.

        **Notes**

        Does not support automatically setting index column(s) similar to how
        ``to_pandas`` works for PyArrow Tables.

        Examples
        --------
        >>> import pyarrow as pa
        >>> import cudf
        >>> data = [pa.array([1, 2, 3]), pa.array([4, 5, 6])]
        >>> batch = pa.RecordBatch.from_arrays(data, ['f0', 'f1'])
        >>> table = pa.Table.from_batches([batch])
        >>> cudf.DataFrame.from_arrow(table)
        <cudf.DataFrame ncols=2 nrows=3 >
        """

        if not isinstance(table, pa.Table):
            raise TypeError("not a pyarrow.Table")

        index_col = None
        dtypes = None
        if isinstance(table.schema.pandas_metadata, dict):
            metadata = table.schema.pandas_metadata
            index_col = metadata["index_columns"]
            dtypes = {
                col["field_name"]: col["pandas_type"]
                for col in metadata["columns"]
                if "field_name" in col
            }

        df = cls()
        for name, col in zip(table.schema.names, table.columns):
            if dtypes:
                dtype = dtypes[name]
                if dtype == "categorical":
                    dtype = "category"
                elif dtype == "date":
                    dtype = "datetime64[ms]"
            else:
                dtype = None

            df[name] = column.as_column(col, dtype=dtype)
        if index_col:
            if isinstance(index_col[0], dict):
                assert index_col[0]["kind"] == "range"
                df = df.set_index(
                    RangeIndex(
                        index_col[0]["start"],
                        index_col[0]["stop"],
                        name=index_col[0]["name"],
                    )
                )
            else:
                df = df.set_index(index_col[0])
                new_index_name = pa.pandas_compat._backwards_compatible_index_name(  # noqa: E501
                    df.index.name, df.index.name
                )
                df.index.name = new_index_name
        return df

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
        members += [(col, self[col].dtype) for col in self.columns]
        dtype = np.dtype(members)
        ret = np.recarray(len(self), dtype=dtype)
        if index:
            ret["index"] = self.index.values
        for col in self.columns:
            ret[col] = self[col].to_array()
        return ret

    @classmethod
    def from_records(self, data, index=None, columns=None, nan_as_null=False):
        """Convert from a numpy recarray or structured array.

        Parameters
        ----------
        data : numpy structured dtype or recarray of ndim=2
        index : str
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
                df[k] = Series(data[:, i], nan_as_null=nan_as_null)
        elif data.ndim == 1:
            for k in names:
                df[k] = Series(data[k], nan_as_null=nan_as_null)

        if index is not None:
            indices = data[index]
            return df.set_index(indices.astype(np.int64))
        return df

    @classmethod
    def from_gpu_matrix(
        self, data, index=None, columns=None, nan_as_null=False
    ):
        """Convert from a numba gpu ndarray.

        Parameters
        ----------
        data : numba gpu ndarray
        index : str
            The name of the index column in *data*.
            If None, the default index is used.
        columns : list of str
            List of column names to include.

        Returns
        -------
        DataFrame
        """
        if data.ndim != 2:
            raise ValueError(
                "matrix dimension expected 2 but found {!r}".format(data.ndim)
            )

        if columns is None:
            names = [i for i in range(data.shape[1])]
        else:
            if len(columns) != data.shape[1]:
                msg = "columns length expected {!r} but found {!r}"
                raise ValueError(msg.format(data.shape[1], len(columns)))
            names = columns

        if index is not None and len(index) != data.shape[0]:
            msg = "index length expected {!r} but found {!r}"
            raise ValueError(msg.format(data.shape[0], len(index)))

        df = DataFrame()
        data = data.transpose()  # to mimic the pandas behaviour
        for i, k in enumerate(names):
            df[k] = Series(data[i], nan_as_null=nan_as_null)

        if index is not None:
            indices = data[index]
            return df.set_index(indices.astype(np.int64))

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

    def _from_columns(cols, index=None, columns=None):
        """
        Construct a DataFrame from a list of Columns
        """
        df = cudf.DataFrame(dict(zip(range(len(cols)), cols)), index=index)
        if columns is not None:
            df.columns = columns
        return df

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
            This  parameter specifies the interpolation method to use,
            when the desired quantile lies between two data points i and j.
            Default 'linear'.
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
            columns = self.columns

        result = DataFrame()

        for k in self.columns:

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

    #
    # Stats
    #
    def _prepare_for_rowwise_op(self):
        """Prepare a DataFrame for CuPy-based row-wise operations.
        """
        warnings.warn(
            "Row-wise operations currently only support int, float, "
            "and bool dtypes."
        )

        if any([col.nullable for col in self._columns]):
            msg = (
                "Row-wise operations do not currently support columns with "
                "null values. Consider removing them with .dropna() "
                "or using .fillna()."
            )
            raise ValueError(msg)

        filtered = self.select_dtypes(include=[np.number, np.bool])
        common_dtype = np.find_common_type(filtered.dtypes, [])
        coerced = filtered.astype(common_dtype)
        return coerced

    def count(self, **kwargs):
        return self._apply_support_method("count", **kwargs)

    def min(self, axis=0, **kwargs):
        return self._apply_support_method("min", axis=axis, **kwargs)

    def max(self, axis=0, **kwargs):
        return self._apply_support_method("max", axis=axis, **kwargs)

    def sum(self, axis=0, **kwargs):
        return self._apply_support_method("sum", axis=axis, **kwargs)

    def product(self, axis=0, **kwargs):
        return self._apply_support_method("prod", axis=axis, **kwargs)

    def prod(self, axis=0, **kwargs):
        """Alias for product.
        """
        return self.product(axis=axis, **kwargs)

    def cummin(self, **kwargs):
        return self._apply_support_method("cummin", **kwargs)

    def cummax(self, **kwargs):
        return self._apply_support_method("cummax", **kwargs)

    def cumsum(self, **kwargs):
        return self._apply_support_method("cumsum", **kwargs)

    def cumprod(self, **kwargs):
        return self._apply_support_method("cumprod", **kwargs)

    def mean(self, axis=0, numeric_only=None, **kwargs):
        """Return the mean of the values for the requested axis.
        Parameters
        ----------
        axis : {index (0), columns (1)}
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
        """
        return self._apply_support_method("mean", axis=axis, **kwargs)

    def std(self, axis=0, ddof=1, **kwargs):
        return self._apply_support_method(
            "std", axis=axis, ddof=ddof, **kwargs
        )

    def var(self, axis=0, ddof=1, **kwargs):
        return self._apply_support_method(
            "var", axis=axis, ddof=ddof, **kwargs
        )

    def kurtosis(self, axis=None, skipna=None, level=None, numeric_only=None):
        if numeric_only not in (None, True):
            msg = "Kurtosis only supports int, float, and bool dtypes."
            raise TypeError(msg)

        self = self.select_dtypes(include=[np.number, np.bool])
        return self._apply_support_method(
            "kurtosis",
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
        )

    def skew(self, axis=None, skipna=None, level=None, numeric_only=None):
        if numeric_only not in (None, True):
            msg = "Skew only supports int, float, and bool dtypes."
            raise TypeError(msg)

        self = self.select_dtypes(include=[np.number, np.bool])
        return self._apply_support_method(
            "skew",
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
        )

    def all(self, bool_only=None, **kwargs):
        if bool_only:
            return self.select_dtypes(include="bool")._apply_support_method(
                "all", **kwargs
            )
        return self._apply_support_method("all", **kwargs)

    def any(self, bool_only=None, **kwargs):
        if bool_only:
            return self.select_dtypes(include="bool")._apply_support_method(
                "any", **kwargs
            )
        return self._apply_support_method("any", **kwargs)

    def _apply_support_method(self, method, axis=0, *args, **kwargs):
        assert axis in (None, 0, 1)

        if axis in (None, 0):
            result = [
                getattr(self[col], method)(*args, **kwargs)
                for col in self._data.keys()
            ]

            if isinstance(result[0], Series):
                support_result = result
                result = DataFrame(index=support_result[0].index)
                for idx, col in enumerate(self._data.keys()):
                    result[col] = support_result[idx]
            else:
                result = Series(result)
                result = result.set_index(self._data.keys())
            return result

        elif axis == 1:
            # for dask metadata compatibility
            skipna = kwargs.pop("skipna", None)
            if skipna not in (None, True, 1):
                msg = (
                    "Row-wise operations do not current support skipna=False."
                )
                raise ValueError(msg)

            prepared = self._prepare_for_rowwise_op()
            arr = cupy.asarray(prepared.as_gpu_matrix())
            result = getattr(arr, method)(axis=1, **kwargs)

            if len(result.shape) == 1:
                return Series(result, index=self.index)
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
            result_columns[col] = self[col]
        return DataFrame(result_columns, index=self.index)

    def select_dtypes(self, include=None, exclude=None):
        """Return a subset of the DataFrame’s columns based on the column dtypes.

        Parameters
        ----------
        include : str or list
            which columns to include based on dtypes
        exclude : str or list
            which columns to exclude based on dtypes

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
                "at least one of include or exclude must be \
                             nonempty"
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
        import cudf.io.parquet as pq

        pq.to_parquet(self, path, *args, **kwargs)

    @ioutils.doc_to_feather()
    def to_feather(self, path, *args, **kwargs):
        """{docstring}"""
        import cudf.io.feather as feather

        feather.to_feather(self, path, *args, **kwargs)

    @ioutils.doc_to_json()
    def to_json(self, path_or_buf=None, *args, **kwargs):
        """{docstring}"""
        import cudf.io.json as json

        json.to_json(self, path_or_buf=path_or_buf, *args, **kwargs)

    @ioutils.doc_to_hdf()
    def to_hdf(self, path_or_buf, key, *args, **kwargs):
        """{docstring}"""
        import cudf.io.hdf as hdf

        hdf.to_hdf(path_or_buf, key, self, *args, **kwargs)

    @ioutils.doc_to_dlpack()
    def to_dlpack(self):
        """{docstring}"""
        import cudf.io.dlpack as dlpack

        return dlpack.to_dlpack(self)

    @ioutils.doc_to_csv()
    def to_csv(
        self,
        path=None,
        sep=",",
        na_rep="",
        columns=None,
        header=True,
        index=True,
        line_terminator="\n",
        chunksize=None,
    ):
        """{docstring}"""
        import cudf.io.csv as csv

        return csv.to_csv(
            self,
            path,
            sep,
            na_rep,
            columns,
            header,
            index,
            line_terminator,
            chunksize,
        )

    @ioutils.doc_to_orc()
    def to_orc(self, fname, compression=None, *args, **kwargs):
        """{docstring}"""
        import cudf.io.orc as orc

        orc.to_orc(self, fname, compression, *args, **kwargs)

    def scatter_by_map(self, map_index, map_size=None, keep_index=True):
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
        # make it a Series
        if isinstance(map_index, str):
            map_index = self[map_index]
        else:
            map_index = Series(map_index)

        # Convert float to integer
        if map_index.dtype == np.float:
            map_index = map_index.astype(np.int32)

        # Convert string or categorical to integer
        if isinstance(map_index._column, StringColumn):
            map_index = Series(
                map_index._column.as_categorical_column(np.int32).as_numerical
            )
            warnings.warn(
                "Using StringColumn for map_index in scatter_by_map. "
                "Use an integer array/column for better performance."
            )
        elif isinstance(map_index._column, CategoricalColumn):
            map_index = Series(map_index._column.as_numerical)
            warnings.warn(
                "Using CategoricalColumn for map_index in scatter_by_map. "
                "Use an integer array/column for better performance."
            )

        if keep_index:
            if isinstance(self.index, cudf.MultiIndex):
                index = self.index.to_frame()._columns
                index_names = self.index.to_frame().columns.to_list()
            else:
                index = [self.index.as_column()]
                index_names = [self.index.name]
        else:
            index = None
            index_names = []

        tables = libcudf.copying.scatter_to_frames(
            self._columns,
            map_index._column,
            index,
            names=self.columns.to_list(),
            index_names=index_names,
        )

        if map_size:
            # Make sure map_size is >= the number of uniques in map_index
            if len(tables) > map_size:
                raise ValueError(
                    "ERROR: map_size must be >= %d (got %d)."
                    % (len(tables), map_size)
                )

            # Append empty dataframes if map_size > len(tables)
            for i in range(map_size - len(tables)):
                tables.append(self.take([]))
        return tables

    def repeat(self, repeats, axis=None):
        assert axis in (None, 0)
        new_index = self.index.repeat(repeats)
        cols = libcudf.filling.repeat(self._columns, repeats)
        # to preserve col names, need to get it from old _cols dict
        column_names = self._data.keys()
        result = DataFrame(data=dict(zip(column_names, cols)))
        return result.set_index(new_index)

    def tile(self, reps):
        """Construct a DataFrame by repeating this DataFrame the number of
        times given by reps

        Parameters
        ----------
        reps : non-negative integer
            The number of repetitions of this DataFrame along axis 0

        Returns
        -------
        The tiled output cudf.DataFrame
        """
        cols = libcudf.filling.tile(self._columns, reps)
        column_names = self._data.keys()
        return DataFrame(data=dict(zip(column_names, cols)))

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
        index_as_cols = self.index.to_frame(index=False)._columns
        new_index_cols = libcudf.filling.repeat(index_as_cols, self.shape[1])
        [last_index] = libcudf.filling.tile(
            [column.as_column(self.columns)], self.shape[0]
        )
        new_index_cols.append(last_index)
        index_df = DataFrame(
            dict(zip(range(0, len(new_index_cols)), new_index_cols))
        )
        new_index = cudf.core.multiindex.MultiIndex.from_frame(index_df)

        # Collect datatypes and cast columns as that type
        common_type = np.result_type(*self.dtypes)
        homogenized_cols = [
            c.astype(common_type)
            if not np.issubdtype(c.dtype, common_type)
            else c
            for c in self._columns
        ]
        data_col = libcudf.reshape.stack(homogenized_cols)
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


def from_pandas(obj):
    """
    Convert certain Pandas objects into the cudf equivalent.

    Supports DataFrame, Series, Index, or MultiIndex.

    Raises
    ------
    TypeError for invalid input type.

    Examples
    --------
    >>> import cudf
    >>> import pandas as pd
    >>> data = [[0, 1], [1, 2], [3, 4]]
    >>> pdf = pd.DataFrame(data, columns=['a', 'b'], dtype=int)
    >>> cudf.from_pandas(pdf)
    <cudf.DataFrame ncols=2 nrows=3 >
    """
    if isinstance(obj, pd.DataFrame):
        return DataFrame.from_pandas(obj)
    elif isinstance(obj, pd.Series):
        return Series.from_pandas(obj)
    elif isinstance(obj, pd.MultiIndex):
        return cudf.MultiIndex.from_pandas(obj)
    elif isinstance(obj, pd.RangeIndex):
        if obj._step and obj._step != 1:
            raise ValueError("cudf RangeIndex requires step == 1")
        return cudf.core.index.RangeIndex(
            obj._start, stop=obj._stop, name=obj.name
        )
    elif isinstance(obj, pd.Index):
        return cudf.Index.from_pandas(obj)
    else:
        raise TypeError(
            "from_pandas only accepts Pandas Dataframes, Series, "
            "Index, RangeIndex and MultiIndex objects. "
            "Got %s" % type(obj)
        )


def merge(left, right, *args, **kwargs):
    return left.merge(right, *args, **kwargs)


# a bit of fanciness to inject doctstring with left parameter
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
