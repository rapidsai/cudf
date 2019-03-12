# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import print_function, division

import inspect
import random
from collections import OrderedDict
from collections.abc import Sequence, Mapping
from copy import copy
import logging
import warnings
import numbers

import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.api.types import is_dict_like

from types import GeneratorType

from librmm_cffi import librmm as rmm
from libgdf_cffi import libgdf

from cudf import formatting, _gdf
from cudf.utils import cudautils, queryutils, applyutils, utils, ioutils
from cudf.dataframe.index import as_index, Index, RangeIndex
from cudf.dataframe.series import Series
from cudf.settings import NOTSET, settings
from cudf.comm.serialize import register_distributed_serializer
from cudf.dataframe.categorical import CategoricalColumn
from cudf.dataframe.buffer import Buffer
from cudf._gdf import nvtx_range_push, nvtx_range_pop
from cudf._sort import get_sorted_inds
from cudf.dataframe import columnops

import cudf.bindings.join as cpp_join


class DataFrame(object):
    """
    A GPU Dataframe object.

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

    Build dataframe with initializer:

    >>> import cudf
    >>> import numpy as np
    >>> from datetime import datetime, timedelta
    >>> ids = np.arange(5)

    Create some datetime data

    >>> t0 = datetime.strptime('2018-10-07 12:00:00', '%Y-%m-%d %H:%M:%S')
    >>> datetimes = [(t0+ timedelta(seconds=x)) for x in range(5)]
    >>> dts = np.array(datetimes, dtype='datetime64')

    Create the GPU DataFrame

    >>> df = cudf.DataFrame([('id', ids), ('datetimes', dts)])
    >>> df
        id                datetimes
    0    0  2018-10-07T12:00:00.000
    1    1  2018-10-07T12:00:01.000
    2    2  2018-10-07T12:00:02.000
    3    3  2018-10-07T12:00:03.000
    4    4  2018-10-07T12:00:04.000

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
    LEFT_RIGHT_INDEX_NAME = 'cudf_left_right_index_key'

    def __init__(self, name_series=None, index=None):
        if index is None:
            index = RangeIndex(start=0)
        self._index = index
        self._size = len(index)
        self._cols = OrderedDict()
        # has initializer?
        if name_series is not None:
            if isinstance(name_series, dict):
                name_series = name_series.items()
            for k, series in name_series:
                self.add_column(k, series, forceindex=index is not None)

    def serialize(self, serialize):
        header = {}
        frames = []
        header['index'], index_frames = serialize(self._index)
        header['index_frame_count'] = len(index_frames)
        frames.extend(index_frames)
        # Use the column directly to avoid duplicating the index
        columns = [col._column for col in self._cols.values()]
        serialized_columns = zip(*map(serialize, columns))
        header['columns'], column_frames = serialized_columns
        header['column_names'] = tuple(self._cols)
        for f in column_frames:
            frames.extend(f)
        return header, frames

    @classmethod
    def deserialize(cls, deserialize, header, frames):
        # Reconstruct the index
        index_header = header['index']
        index_frames = frames[:header['index_frame_count']]
        index = deserialize(index_header, index_frames)
        # Reconstruct the columns
        column_frames = frames[header['index_frame_count']:]
        columns = []
        for k, meta in zip(header['column_names'], header['columns']):
            col_frame_count = meta['frame_count']
            colobj = deserialize(meta, column_frames[:col_frame_count])
            columns.append((k, colobj))
            # Advance frames
            column_frames = column_frames[col_frame_count:]
        return cls(columns, index=index)

    @property
    def dtypes(self):
        """Return the dtypes in this object."""
        return pd.Series([x.dtype for x in self._cols.values()],
                         index=self._cols.keys())

    @property
    def shape(self):
        """Returns a tuple representing the dimensionality of the DataFrame.
        """
        return len(self), len(self._cols)

    def __dir__(self):
        o = set(dir(type(self)))
        o.update(self.__dict__)
        o.update(c for c in self.columns if
                 (isinstance(c, pd.compat.string_types) and
                  pd.compat.isidentifier(c)))
        return list(o)

    def __getattr__(self, key):
        if key != '_cols' and key in self._cols:
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
        if isinstance(arg, str) or isinstance(arg, numbers.Integral) or \
           isinstance(arg, tuple):
            s = self._cols[arg]
            s.name = arg
            return s
        elif isinstance(arg, slice):
            df = DataFrame()
            for k, col in self._cols.items():
                df[k] = col[arg]
            return df
        elif isinstance(arg, (list, np.ndarray, pd.Series,
                        Series, Index, pd.Index)):
            mask = arg
            if isinstance(mask, list):
                mask = np.array(mask)
            df = DataFrame()
            if(mask.dtype == 'bool'):
                # New df-wide index
                selvals, selinds = columnops.column_select_by_boolmask(
                        columnops.as_column(self.index), Series(mask))
                index = self.index.take(selinds.to_gpu_array())
                for col in self._cols:
                    df[col] = Series(self._cols[col][arg], index=index)
                df.set_index(index)
            else:
                for col in arg:
                    df[col] = self[col]
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
                boolbits = cudautils.compact_mask_bytes(
                           other[col].to_gpu_array())
            else:
                boolbits = cudautils.make_empty_mask(len(self[col]))
            df[col]._column = df[col]._column.set_mask(boolbits)
        return df

    def __setitem__(self, name, col):
        """Add/set column by *name or DataFrame*
        """
        # div[div < 0] = 0
        if isinstance(name, DataFrame):
            for col_name in self._cols:
                mask = name[col_name]
                self._cols[col_name] = self._cols[col_name] \
                                           .masked_assign(value=col, mask=mask)

        elif name in self._cols:
            self._cols[name] = self._prepare_series_for_add(col)
        else:
            self.add_column(name, col)

    def __delitem__(self, name):
        """
        Drop the given column by *name*.
        """
        self._drop_column(name)

    def __sizeof__(self):
        return sum(col.__sizeof__() for col in self._cols.values())

    def __len__(self):
        """
        Returns the number of rows
        """
        return self._size

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__' and 'sqrt' == ufunc.__name__:
            from cudf import sqrt
            return sqrt(self)
        else:
            return NotImplemented

    @property
    def empty(self):
        return not len(self)

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

    def to_string(self, nrows=NOTSET, ncols=NOTSET):
        """
        Convert to string

        Parameters
        ----------
        nrows : int
            Maximum number of rows to show.
            If it is None, all rows are shown.

        ncols : int
            Maximum number of columns to show.
            If it is None, all columns are shown.

        Examples
        --------
        >>> import cudf
        >>> df = cudf.DataFrame()
        >>> df['key'] = [0, 1, 2]
        >>> df['val'] = [float(i + 10) for i in range(3)]
        >>> df.to_string()
        '   key   val\\n0    0  10.0\\n1    1  11.0\\n2    2  12.0'
        """
        if nrows is NOTSET:
            nrows = settings.formatting.get('nrows')
        if ncols is NOTSET:
            ncols = settings.formatting.get('ncols')

        if nrows is None:
            nrows = len(self)
        else:
            nrows = min(nrows, len(self))  # cap row count

        if ncols is None:
            ncols = len(self.columns)
        else:
            ncols = min(ncols, len(self.columns))  # cap col count

        more_cols = len(self.columns) - ncols
        more_rows = len(self) - nrows

        # Prepare cells
        cols = OrderedDict()
        use_cols = list(self.columns[:ncols - 1])
        if ncols > 0:
            use_cols.append(self.columns[-1])

        for h in use_cols:
            cols[h] = self[h].values_to_string(nrows=nrows)

        # Format into a table
        return formatting.format(index=self._index, cols=cols,
                                 show_headers=True, more_cols=more_cols,
                                 more_rows=more_rows, min_width=2)

    def __str__(self):
        nrows = settings.formatting.get('nrows') or 10
        ncols = settings.formatting.get('ncols') or 8
        return self.to_string(nrows=nrows, ncols=ncols)

    def __repr__(self):
        return "<cudf.DataFrame ncols={} nrows={} >".format(
            len(self.columns),
            len(self),
        )

    # binary, rbinary, unary, orderedcompare, unorderedcompare
    def _call_op(self, other, internal_fn, fn):
        result = DataFrame()
        result.set_index(self.index)
        if internal_fn == '_unaryop':
            for col in self._cols:
                result[col] = self._cols[col]._unaryop(fn)
        elif isinstance(other, Sequence):
            for k, col in enumerate(self._cols):
                result[col] = getattr(self._cols[col], internal_fn)(
                        other[k],
                        fn,
                )
        elif isinstance(other, DataFrame):
            for col in other._cols:
                if col in self._cols:
                    result[col] = getattr(self._cols[col], internal_fn)(
                            other._cols[col],
                            fn,
                    )
                else:
                    result[col] = Series(cudautils.full(self.shape[0],
                                         np.dtype('float64').type(np.nan),
                                         'float64'), nan_as_null=False)
            for col in self._cols:
                if col not in other._cols:
                    result[col] = Series(cudautils.full(self.shape[0],
                                         np.dtype('float64').type(np.nan),
                                         'float64'), nan_as_null=False)
        elif isinstance(other, Series):
            raise NotImplementedError(
                    "Series to DataFrame arithmetic not supported "
                    "until strings can be used as indices. Try converting your"
                    " Series into a DataFrame first.")
        elif isinstance(other, numbers.Number):
            for col in self._cols:
                result[col] = getattr(self._cols[col], internal_fn)(
                        other,
                        fn,
                )
        else:
            raise NotImplementedError(
                    "DataFrame operations with " + str(type(other)) + " not "
                    "supported at this time.")
        return result

    def _binaryop(self, other, fn):
        return self._call_op(other, '_binaryop', fn)

    def _rbinaryop(self, other, fn):
        return self._call_op(other, '_rbinaryop', fn)

    def _unaryop(self, fn):
        return self._call_op(self, '_unaryop', fn)

    def __add__(self, other):
        return self._binaryop(other, 'add')

    def __radd__(self, other):
        return self._rbinaryop(other, 'add')

    def __sub__(self, other):
        return self._binaryop(other, 'sub')

    def __rsub__(self, other):
        return self._rbinaryop(other, 'sub')

    def __mul__(self, other):
        return self._binaryop(other, 'mul')

    def __rmul__(self, other):
        return self._rbinaryop(other, 'mul')

    def __pow__(self, other):
        if other == 2:
            return self * self
        else:
            return NotImplemented

    def __floordiv__(self, other):
        return self._binaryop(other, 'floordiv')

    def __rfloordiv__(self, other):
        return self._rbinaryop(other, 'floordiv')

    def __truediv__(self, other):
        return self._binaryop(other, 'truediv')

    def __rtruediv__(self, other):
        return self._rbinaryop(other, 'truediv')

    __div__ = __truediv__

    def _unordered_compare(self, other, cmpops):
        return self._call_op(other, '_unordered_compare', cmpops)

    def _ordered_compare(self, other, cmpops):
        return self._call_op(other, '_ordered_compare', cmpops)

    def __eq__(self, other):
        return self._unordered_compare(other, 'eq')

    def __ne__(self, other):
        return self._unordered_compare(other, 'ne')

    def __lt__(self, other):
        return self._ordered_compare(other, 'lt')

    def __le__(self, other):
        return self._ordered_compare(other, 'le')

    def __gt__(self, other):
        return self._ordered_compare(other, 'gt')

    def __ge__(self, other):
        return self._ordered_compare(other, 'ge')

    def __iter__(self):
        return iter(self.columns)

    def iteritems(self):
        """ Iterate over column names and series pairs """
        for k in self:
            yield (k, self[k])

    @property
    def loc(self):
        """
        Returns a label-based indexer for row-slicing and column selection.

        Examples
        --------
        >>> df = DataFrame([('a', list(range(20))),
        ...                 ('b', list(range(20))),
        ...                 ('c', list(range(20)))])

        Get the row by index label from 'a' and 'b' columns

        >>> df.loc[0, ['a', 'b']]
        a    0
        b    0

        Get rows from index 2 to index 5 from 'a' and 'b' columns.

        >>> df.loc[2:5, ['a', 'b']]
           a  b
        2  2  2
        3  3  3
        4  4  4
        5  5  5

        Get the every 3rd rows from index 2 to 10 from 'a' and 'b'

        >>> df.loc[2:10:3, ['a', 'b']]
            a    b
        2   2    2
        5   5    5
        8   8    8
        """
        return Loc(self)

    @property
    def iloc(self):
        """
        Returns a  integer-location based indexer for selection by position.

        Examples
        --------
        >>> df = DataFrame([('a', list(range(20))),
        ...                 ('b', list(range(20))),
        ...                 ('c', list(range(20)))])
        >>> df.iloc[1]  # get the row from index 1st
        a    1
        b    1
        c    1
        >>> df.iloc[[0, 2, 9, 18]]  # get the rows from indices 0,2,9 and 18.
              a    b    c
         0    0    0    0
         2    2    2    2
         9    9    9    9
        18   18   18   18
        >>> df.iloc[3:10:2]  # get the rows using slice indices
             a    b    c
        3    3    3    3
        5    5    5    5
        7    7    7    7
        9    9    9    9
        """
        return Iloc(self)

    @property
    def columns(self):
        """Returns a tuple of columns
        """
        return pd.Index(self._cols)

    @columns.setter
    def columns(self, columns):
        old_cols = list(self._cols.keys())
        l_old_cols = len(old_cols)
        l_new_cols = len(columns)
        if l_new_cols != l_old_cols:
            msg = f'Length of new column names: {l_new_cols} does not ' \
                  'match length of previous column names: {l_old_cols}'
            raise ValueError(msg)

        mapper = dict(zip(old_cols, columns))
        self.rename(mapper=mapper, inplace=True)

    @property
    def index(self):
        """Returns the index of the DataFrame
        """
        return self._index

    @index.setter
    def index(self, _index):
        new_length = len(_index)
        old_length = len(self._index)

        if new_length != old_length:
            msg = f'Length mismatch: Expected index has {old_length}' \
                    ' elements, new values have {new_length} elements'
            raise ValueError(msg)

        # try to build an index from generic _index
        idx = as_index(_index)
        self._index = idx
        for k in self.columns:
            self[k] = self[k].set_index(idx)

    def set_index(self, index):
        """Return a new DataFrame with a new index

        Parameters
        ----------
        index : Index, Series-convertible, or str
            Index : the new index.
            Series-convertible : values for the new index.
            str : name of column to be used as series
        """
        # When index is a column name
        if isinstance(index, str):
            df = self.copy(deep=False)
            df._drop_column(index)
            return df.set_index(self[index])
        # Otherwise
        else:
            index = index if isinstance(index, Index) else as_index(index)
            df = DataFrame()
            df._index = index
            for k in self.columns:
                df[k] = self[k].set_index(index)
            return df

    def reset_index(self, drop=False):
        if not drop:
            name = self.index.name or 'index'
            out = DataFrame()
            out[name] = self.index
            for c in self.columns:
                out[c] = self[c]
        else:
            out = self
        return out.set_index(RangeIndex(len(self)))

    def take(self, positions, ignore_index=False):
        out = DataFrame()
        for col in self.columns:
            out[col] = self[col].take(positions, ignore_index=ignore_index)
        return out

    def copy(self, deep=True):
        """
        Returns a copy of this dataframe

        Parameters
        ----------
        deep: bool
           Make a full copy of Series columns and Index at the GPU level, or
           create a new allocation with references.
        """
        df = DataFrame()
        df._size = self._size
        if deep:
            df._index = self._index.copy(deep)
            for k in self._cols:
                df._cols[k] = self._cols[k].copy(deep)
        else:
            df._index = self._index
            for k in self._cols:
                df._cols[k] = self._cols[k]
        return df

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

    def _sanitize_columns(self, col):
        """Sanitize pre-appended
           col values
        """
        series = Series(col)
        if len(self) == 0 and len(self.columns) > 0 and len(series) > 0:
            ind = series.index
            arr = rmm.device_array(shape=len(ind), dtype=np.float64)
            size = utils.calc_chunk_size(arr.size, utils.mask_bitsize)
            mask = cudautils.zeros(size, dtype=utils.mask_dtype)
            val = Series.from_masked_array(arr, mask, null_count=len(ind))
            for name in self._cols:
                self._cols[name] = val
            self._index = series.index
            self._size = len(series)

    def _sanitize_values(self, col):
        """Sanitize col values before
           being added
        """
        index = self._index
        series = Series(col)
        sind = series.index

        # This won't handle 0 dimensional arrays which should be okay
        SCALAR = np.isscalar(col)

        if len(self) > 0 and len(series) == 1 and SCALAR:
            arr = rmm.device_array(shape=len(index), dtype=series.dtype)
            cudautils.gpu_fill_value.forall(arr.size)(arr, col)
            return Series(arr)
        elif len(self) > 0 and len(sind) != len(index):
            raise ValueError('Length of values does not match index length')
        return col

    def _prepare_series_for_add(self, col, forceindex=False):
        """Prepare a series to be added to the DataFrame.

        Parameters
        ----------
        col : Series, array-like
            Values to be added.

        Returns
        -------
        The prepared Series object.
        """
        self._sanitize_columns(col)
        col = self._sanitize_values(col)

        empty_index = len(self._index) == 0
        series = Series(col)
        if forceindex or empty_index or self._index.equals(series.index):
            if empty_index:
                self._index = series.index
            self._size = len(series)
            return series
        else:
            return series.set_index(self._index)

    def add_column(self, name, data, forceindex=False):
        """Add a column

        Parameters
        ----------
        name : str
            Name of column to be added.
        data : Series, array-like
            Values to be added.
        """

        if name in self._cols:
            raise NameError('duplicated column name {!r}'.format(name))

        if isinstance(data, GeneratorType):
            data = Series(data)
        series = self._prepare_series_for_add(data, forceindex=forceindex)
        series.name = name
        self._cols[name] = series

    def drop(self, labels):
        """Drop column(s)

        Parameters
        ----------
        labels : str or sequence of strings
            Name of column(s) to be dropped.

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
        columns = [labels] if isinstance(labels, str) else list(labels)

        outdf = self.copy()
        for c in columns:
            outdf._drop_column(c)
        return outdf

    def drop_column(self, name):
        """Drop a column by *name*
        """
        warnings.warn(
                'The drop_column method is deprecated. '
                'Use the drop method instead.',
                DeprecationWarning
            )
        self._drop_column(name)

    def _drop_column(self, name):
        """Drop a column by *name*
        """
        if name not in self._cols:
            raise NameError('column {!r} does not exist'.format(name))
        del self._cols[name]

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
            Retrun new DataFrame.  If True, assign columns without copy

        Returns
        -------
        DataFrame

        Notes
        -----
        Difference from pandas:
          * Support axis='columns' only.
          * Not supporting: index, level
        """
        # Pandas defaults to using columns over mapper
        if columns:
            mapper = columns

        out = DataFrame()
        out = out.set_index(self.index)

        if isinstance(mapper, Mapping):
            for column in self.columns:
                if column in mapper:
                    out[mapper[column]] = self[column]
                else:
                    out[column] = self[column]
        elif callable(mapper):
            for column in self.columns:
                out[mapper(column)] = self[column]

        if inplace:
            self._cols = out._cols
        else:
            return out.copy(deep=copy)

    @classmethod
    def _concat(cls, objs, axis=0, ignore_index=False):
        nvtx_range_push("CUDF_CONCAT", "orange")
        if len(set(frozenset(o.columns) for o in objs)) != 1:
            what = set(frozenset(o.columns) for o in objs)
            raise ValueError('columns mismatch: {}'.format(what))

        objs = [o for o in objs]
        if ignore_index:
            index = RangeIndex(sum(map(len, objs)))
        else:
            index = Index._concat([o.index for o in objs])
        data = [(c, Series._concat([o[c] for o in objs], index=index))
                for c in objs[0].columns]
        out = cls(data)
        out._index = index
        nvtx_range_pop()
        return out

    def as_gpu_matrix(self, columns=None, order='F'):
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

        cols = [self._cols[k] for k in columns]
        ncol = len(cols)
        nrow = len(self)
        if ncol < 1:
            raise ValueError("require at least 1 column")
        if nrow < 1:
            raise ValueError("require at least 1 row")
        dtype = cols[0].dtype
        if any(dtype != c.dtype for c in cols):
            raise ValueError('all columns must have the same dtype')
        for k, c in self._cols.items():
            if c.null_count > 0:
                errmsg = ("column {!r} has null values. "
                          "hint: use .fillna() to replace null values")
                raise ValueError(errmsg.format(k))

        if order == 'F':
            matrix = rmm.device_array(shape=(nrow, ncol), dtype=dtype,
                                      order=order)
            for colidx, inpcol in enumerate(cols):
                dense = inpcol.to_gpu_array(fillna='pandas')
                matrix[:, colidx].copy_to_device(dense)
        elif order == 'C':
            matrix = cudautils.row_matrix(cols, nrow, ncol, dtype)
        else:
            errmsg = ("order parameter should be 'C' for row major or 'F' for"
                      "column major GPU matrix")
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

    def one_hot_encoding(self, column, prefix, cats, prefix_sep='_',
                         dtype='float64'):
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
        newnames = [prefix_sep.join([prefix, str(cat)]) for cat in cats]
        newcols = self[column].one_hot_encoding(cats=cats, dtype=dtype)
        outdf = self.copy()
        for name, col in zip(newnames, newcols):
            outdf.add_column(name, col)
        return outdf

    def label_encoding(self, column, prefix, cats, prefix_sep='_', dtype=None,
                       na_sentinel=-1):
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

        newname = prefix_sep.join([prefix, 'labels'])
        newcol = self[column].label_encoding(cats=cats, dtype=dtype,
                                             na_sentinel=na_sentinel)
        outdf = self.copy()
        outdf.add_column(newname, newcol)

        return outdf

    def _sort_by(self, sorted_indices):
        df = DataFrame()
        # Perform out = data[index] for all columns
        for k in self.columns:
            df[k] = self[k].take(sorted_indices.to_gpu_array())
        return df

    def argsort(self, ascending=True, na_position='last'):
        cols = [series._column for series in self._cols.values()]
        return get_sorted_inds(cols, ascending=ascending,
                               na_position=na_position)

    def sort_index(self, ascending=True):
        """Sort by the index
        """
        return self._sort_by(self.index.argsort(ascending=ascending))

    def sort_values(self, by, ascending=True, na_position='last'):
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
        return self._sort_by(self[by].argsort(
            ascending=ascending,
            na_position=na_position)
        )

    def nlargest(self, n, columns, keep='first'):
        """Get the rows of the DataFrame sorted by the n largest value of *columns*

        Notes
        -----
        Difference from pandas:
        * Only a single column is supported in *columns*
        """
        return self._n_largest_or_smallest('nlargest', n, columns, keep)

    def nsmallest(self, n, columns, keep='first'):
        """Get the rows of the DataFrame sorted by the n smallest value of *columns*

        Difference from pandas:
        * Only a single column is supported in *columns*
        """
        return self._n_largest_or_smallest('nsmallest', n, columns, keep)

    def _n_largest_or_smallest(self, method, n, columns, keep):
        # Get column to operate on
        if not isinstance(columns, str):
            [column] = columns
        else:
            column = columns
        if not (0 <= n < len(self)):
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
        if len(self.columns) == 0:
            return self

        dtype = self.dtypes[0]
        if pd.api.types.is_categorical_dtype(dtype):
            raise NotImplementedError('Categorical columns are not yet '
                                      'supported for function')
        if any(t != dtype for t in self.dtypes):
            raise ValueError('all columns must have the same dtype')
        has_null = any(c.null_count for c in self._cols.values())

        df = DataFrame()

        ncols = len(self.columns)
        cols = [self[col]._column.cffi_view for col in self._cols]

        new_nrow = ncols
        new_ncol = len(self)

        if has_null:
            new_col_series = [
                Series.from_masked_array(
                    data=Buffer(rmm.device_array(shape=new_nrow, dtype=dtype)),
                    mask=cudautils.make_empty_mask(size=new_nrow),
                )
                for i in range(0, new_ncol)]
        else:
            new_col_series = [
                Series(
                    data=Buffer(rmm.device_array(shape=new_nrow, dtype=dtype)),
                )
                for i in range(0, new_ncol)]
        new_col_ptrs = [
            new_col_series[i]._column.cffi_view
            for i in range(0, new_ncol)]

        # TODO (dm): move to _gdf.py
        libgdf.gdf_transpose(
            ncols,
            cols,
            new_col_ptrs
        )

        for series in new_col_series:
            series._column._update_null_count()

        for i in range(0, new_ncol):
            df[str(i)] = new_col_series[i]
        return df

    @property
    def T(self):
        return self.transpose()

    def merge(self, right, on=None, how='inner', left_on=None, right_on=None,
              left_index=False, right_index=False, lsuffix=None, rsuffix=None,
              type="", method='hash', indicator=False, suffixes=('_x', '_y')):
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
        how : str, defaults to 'left'
            Only accepts 'left'
            left: use only keys from left frame, similar to
            a SQL left outer join; preserve key order
        suffixes: Tuple[str, str], defaults to ('_x', '_y')
            Suffixes applied to overlapping column names on the left and right
            sides
        type : str, defaults to 'hash'

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
        _gdf.nvtx_range_push("CUDF_JOIN", "blue")
        if indicator:
            raise NotImplementedError(
                "Only indicator=False is currently supported"
            )

        if lsuffix or rsuffix:
            raise ValueError(
                "The lsuffix and rsuffix keywords have been replaced with the "
                "``suffixes=`` keyword.  "
                "Please provide the following instead: \n\n"
                "    suffixes=('%s', '%s')" %
                (lsuffix or '_x', rsuffix or '_y')
            )
        else:
            lsuffix, rsuffix = suffixes

        if left_on and right_on:
            raise NotImplementedError("left_on='x', right_on='y' not supported"
                                      "in CUDF at this time.")

        lhs = self.copy(deep=False)
        rhs = right.copy(deep=False)
        if on:
            on = copy(on)
        if left_on:
            left_on = copy(left_on)
        if right_on:
            right_on = copy(right_on)

        # Early termination Error checking
        if type != "":
            warnings.warn(
                'type="' + type + '" parameter is deprecated.'
                'Use method="' + type + '" instead.',
                DeprecationWarning
            )
            method = type
        if how not in ['left', 'inner', 'outer']:
            raise NotImplementedError('{!r} merge not supported yet'
                                      .format(how))
        same_names = set(lhs.columns) & set(rhs.columns)
        if same_names and not (lsuffix or rsuffix):
            raise ValueError('there are overlapping columns but '
                             'lsuffix and rsuffix are not defined')

        def fix_name(name, suffix):
            if name in same_names:
                return "{}{}".format(name, suffix)
            return name

        if left_index and right_index:
            on = lhs.LEFT_RIGHT_INDEX_NAME
            lhs[on] = lhs.index
            rhs[on] = rhs.index
        if on is None and left_on is None and right_on is None:
            on = list(same_names)
            if len(on) == 0:
                raise ValueError('No common columns to perform merge on')

        # Essential parameters
        if on:
            on = [on] if isinstance(on, str) else list(on)

        # Pandas inconsistency warning
        if len(lhs) == 0 and len(lhs.columns) > len(rhs.columns) and\
                set(rhs.columns).intersection(lhs.columns):
            logging.warning(
                    "Pandas and CUDF column ordering may not match for "
                    "DataFrames with 0 rows."
                    )

        # Column prep - this should be simplified
        col_cats = {}

        for name in left_on or []:
            if pd.api.types.is_categorical_dtype(lhs[name]):
                lcats = lhs[name].cat.categories
                rcats = rhs[name].cat.categories
                if how == 'rhs':
                    cats = rcats
                    lhs[name] = (lhs[name].cat._set_categories(cats)
                                 .fillna(-1))
                elif how in ['inner', 'outer']:
                    # Do the join using the union of categories from both side.
                    # Adjust for inner joins afterwards
                    cats = sorted(set(lcats) | set(rcats))
                    lhs[name] = (lhs[name].cat._set_categories(cats)
                                 .fillna(-1))
                    lhs[name] = lhs[name]._column.as_numerical
                    rhs[name] = (rhs[name].cat._set_categories(cats)
                                 .fillna(-1))
                    rhs[name] = rhs[name]._column.as_numerical
                col_cats[name] = cats
        for name in right_on or []:
            if pd.api.types.is_categorical_dtype(rhs[name]):
                lcats = lhs[name].cat.categories
                rcats = rhs[name].cat.categories
                if how == 'left':
                    cats = lcats
                    rhs[name] = (rhs[name].cat._set_categories(cats)
                                 .fillna(-1))
                elif how in ['inner', 'outer']:
                    # Do the join using the union of categories from both side.
                    # Adjust for inner joins afterwards
                    cats = sorted(set(lcats) | set(rcats))
                    lhs[name] = (lhs[name].cat._set_categories(cats)
                                 .fillna(-1))
                    lhs[name] = lhs[name]._column.as_numerical
                    rhs[name] = (rhs[name].cat._set_categories(cats)
                                 .fillna(-1))
                    rhs[name] = rhs[name]._column.as_numerical
                col_cats[name] = cats
        for name, col in lhs._cols.items():
            if pd.api.types.is_categorical_dtype(col) and name not in on:
                f_n = fix_name(name, lsuffix)
                col_cats[f_n] = lhs[name].cat.categories
        for name, col in rhs._cols.items():
            if pd.api.types.is_categorical_dtype(col) and name not in on:
                f_n = fix_name(name, rsuffix)
                col_cats[f_n] = rhs[name].cat.categories

        if right_on and left_on:
            raise NotImplementedError("merge(left_on='x', right_on='y' not"
                                      "supported by CUDF at this time.")
        if left_index and right_on:
            lhs[right_on] = lhs.index
            left_on = right_on
        elif right_index and left_on:
            rhs[left_on] = rhs.index
            right_on = left_on

        if on:
            left_on = on
            right_on = on

        # Compute merge
        cols, valids = cpp_join.join(lhs._cols, rhs._cols, left_on, right_on,
                                     how, method=method)

        # Output conversion - take cols and valids from `cpp_join` and
        # combine into a DataFrame()
        df = DataFrame()

        # Columns are returned in order on - left - rhs from libgdf
        # In order to mirror pandas, reconstruct our df using the
        # columns from `left` and the data from `cpp_join`. The final order
        # is left columns, followed by non-join-key rhs columns.
        on_count = 0
        on = list(set(right_on + left_on))
        # gap spaces between left and `on` for result from `cpp_join`
        gap = len(lhs.columns) - len(on)
        for idc, name in enumerate(lhs.columns):
            if name in on:
                # on columns returned first from `cpp_join`
                for idx in range(len(on)):
                    if on[idx] == name:
                        on_idx = idx + gap
                        on_count = on_count + 1
                        key = on[idx]
                        categories = col_cats[key] if key in col_cats.keys()\
                            else None
                        df[key] = columnops.build_column(
                                Buffer(cols[on_idx]),
                                dtype=cols[on_idx].dtype,
                                mask=Buffer(valids[on_idx]),
                                categories=categories,
                                )
            else:  # not an `on`-column, `cpp_join` returns these after `on`
                # but they need to be added to the result before `on` columns.
                # on_count corrects gap for non-`on` columns
                left_column_idx = idc - on_count
                left_name = fix_name(name, lsuffix)
                categories = col_cats[left_name] if left_name in\
                    col_cats.keys() else None
                df[left_name] = columnops.build_column(
                        Buffer(cols[left_column_idx]),
                        dtype=cols[left_column_idx].dtype,
                        mask=Buffer(valids[left_column_idx]),
                        categories=categories,
                        )
        rhs_column_idx = len(lhs.columns)
        for name in rhs.columns:
            if name not in on:
                # now copy the columns from `rhs` that were not in `on`
                rhs_name = fix_name(name, rsuffix)
                categories = col_cats[rhs_name] if rhs_name in\
                    col_cats.keys() else None
                df[rhs_name] = columnops.build_column(
                        Buffer(cols[rhs_column_idx]),
                        dtype=cols[rhs_column_idx].dtype,
                        mask=Buffer(valids[rhs_column_idx]),
                        categories=categories,
                        )
                rhs_column_idx = rhs_column_idx + 1

        if left_index and right_index:
            df = df.drop(lhs.LEFT_RIGHT_INDEX_NAME)
            df = df.set_index(lhs.index[df.index.gpu_values])
        elif right_index and left_on:
            new_index = Series(lhs.index,
                               index=RangeIndex(0, len(lhs[left_on])))
            indexed = lhs[left_on][df[left_on]-1]
            new_index = new_index[indexed-1]
            df.index = new_index
        elif left_index and right_on:
            new_index = Series(rhs.index,
                               index=RangeIndex(0, len(rhs[right_on])))
            indexed = rhs[right_on][df[right_on]-1]
            new_index = new_index[indexed-1]
            df.index = new_index

        _gdf.nvtx_range_pop()

        return df

    def join(self, other, on=None, how='left', lsuffix='', rsuffix='',
             sort=False, type="", method='hash'):
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

        _gdf.nvtx_range_push("CUDF_JOIN", "blue")

        # Outer joins still use the old implementation
        if type != "":
            warnings.warn(
                'type="' + type + '" parameter is deprecated.'
                'Use method="' + type + '" instead.',
                DeprecationWarning
            )
            method = type

        if how not in ['left', 'right', 'inner', 'outer']:
            raise NotImplementedError('unsupported {!r} join'.format(how))

        if how == 'right':
            # libgdf doesn't support right join directly, we will swap the
            # dfs and use left join
            return other.join(self, other, how='left', lsuffix=rsuffix,
                              rsuffix=lsuffix, sort=sort, method='hash')

        same_names = set(self.columns) & set(other.columns)
        if same_names and not (lsuffix or rsuffix):
            raise ValueError('there are overlapping columns but '
                             'lsuffix and rsuffix are not defined')

        lhs = DataFrame()
        rhs = DataFrame()

        # Creating unique column name to use libgdf join
        idx_col_name = str(random.randint(2**29, 2**31))

        while idx_col_name in self.columns or idx_col_name in other.columns:
            idx_col_name = str(random.randint(2**29, 2**31))

        lhs[idx_col_name] = Series(self.index.as_column()).set_index(self
                                                                     .index)
        rhs[idx_col_name] = Series(other.index.as_column()).set_index(other
                                                                      .index)

        for name in self.columns:
            lhs[name] = self[name]

        for name in other.columns:
            rhs[name] = other[name]

        lhs = lhs.reset_index(drop=True)
        rhs = rhs.reset_index(drop=True)

        cat_join = False

        if pd.api.types.is_categorical_dtype(lhs[idx_col_name]):
            cat_join = True
            lcats = lhs[idx_col_name].cat.categories
            rcats = rhs[idx_col_name].cat.categories
            if how == 'left':
                cats = lcats
                rhs[idx_col_name] = (rhs[idx_col_name].cat
                                                      ._set_categories(cats)
                                                      .fillna(-1))
            elif how == 'right':
                cats = rcats
                lhs[idx_col_name] = (lhs[idx_col_name].cat
                                                      ._set_categories(cats)
                                                      .fillna(-1))
            elif how in ['inner', 'outer']:
                cats = sorted(set(lcats) | set(rcats))

                lhs[idx_col_name] = (lhs[idx_col_name].cat
                                                      ._set_categories(cats)
                                                      .fillna(-1))
                lhs[idx_col_name] = lhs[idx_col_name]._column.as_numerical

                rhs[idx_col_name] = (rhs[idx_col_name].cat
                                                      ._set_categories(cats)
                                                      .fillna(-1))
                rhs[idx_col_name] = rhs[idx_col_name]._column.as_numerical

        if lsuffix == '':
            lsuffix = 'l'
        if rsuffix == '':
            rsuffix = 'r'

        df = lhs.merge(rhs, on=[idx_col_name], how=how,
                       suffixes=(lsuffix, rsuffix), method=method)

        if cat_join:
            df[idx_col_name] = CategoricalColumn(data=df[idx_col_name].data,
                                                 categories=cats,
                                                 ordered=False)

        df = df.set_index(idx_col_name)

        if sort and len(df):
            return df.sort_index()

        return df

    def groupby(self, by=None, sort=False, as_index=True, method="hash",
                level=None):
        """Groupby

        Parameters
        ----------
        by : list-of-str or str
            Column name(s) to form that groups by.
        sort : bool
            Force sorting group keys.
            Depends on the underlying algorithm.
        as_index : bool; defaults to False
            Must be False.  Provided to be API compatible with pandas.
            The keys are always left as regular columns in the result.
        method : str, optional
            A string indicating the method to use to perform the group by.
            Valid values are "hash" or "cudf".
            "cudf" method may be deprecated in the future, but is currently
            the only method supporting group UDFs via the `apply` function.

        Returns
        -------
        The groupby object

        Notes
        -----
        Unlike pandas, this groupby operation behaves like a SQL groupby.
        No empty rows are returned.  (For categorical keys, pandas returns
        rows for all categories even if they are no corresponding values.)

        Only a minimal number of operations is implemented so far.

        - Only *by* argument is supported.
        - Since we don't support multiindex, the *by* columns are stored
          as regular columns.
        """

        if by is None and level is None:
            raise TypeError('groupby() requires either by or level to be'
                            'specified.')
        if (method == "cudf"):
            from cudf.groupby.legacy_groupby import Groupby
            if as_index:
                warnings.warn(
                    'as_index==True not supported due to the lack of '
                    'multi-index with legacy groupby function. Use hash '
                    'method for multi-index'
                )
            result = Groupby(self, by=by)
            return result
        else:
            from cudf.groupby.groupby import Groupby

            _gdf.nvtx_range_push("CUDF_GROUPBY", "purple")
            # The matching `pop` for this range is inside LibGdfGroupby
            # __apply_agg
            result = Groupby(self, by=by, method=method, as_index=as_index,
                             level=level)
            return result

    def query(self, expr):
        """
        Query with a boolean expression using Numba to compile a GPU kernel.

        See pandas.DataFrame.query.

        Parameters
        ----------

        expr : str
            A boolean expression. Names in expression refer to columns.

            Names starting with `@` refer to Python variables

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
        """
        _gdf.nvtx_range_push("CUDF_QUERY", "purple")
        # Get calling environment
        callframe = inspect.currentframe().f_back
        callenv = {
            'locals': callframe.f_locals,
            'globals': callframe.f_globals,
        }
        # Run query
        boolmask = queryutils.query_execute(self, expr, callenv)

        selected = Series(boolmask)
        newdf = DataFrame()
        for col in self.columns:
            newseries = self[col][selected]
            newdf[col] = newseries
        result = newdf
        _gdf.nvtx_range_pop()
        return result

    @applyutils.doc_apply()
    def apply_rows(self, func, incols, outcols, kwargs, cache_key=None):
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
        return applyutils.apply_rows(self, func, incols, outcols, kwargs,
                                     cache_key=cache_key)

    @applyutils.doc_applychunks()
    def apply_chunks(self, func, incols, outcols, kwargs={}, chunks=None,
                     tpb=1):
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
            raise ValueError('*chunks* must be defined')
        return applyutils.apply_chunks(self, func, incols, outcols, kwargs,
                                       chunks=chunks, tpb=tpb)

    def hash_columns(self, columns=None):
        """Hash the given *columns* and return a new Series

        Parameters
        ----------
        column : sequence of str; optional
            Sequence of column names. If columns is *None* (unspecified),
            all columns in the frame are used.
        """
        from cudf.dataframe import numerical

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
        cols = [col._column for col in self._cols.values()]
        names = list(self._cols.keys())
        key_indices = [names.index(k) for k in columns]
        # Allocate output buffers
        outputs = [col.copy() for col in cols]
        # Call hash_partition
        offsets = _gdf.hash_partition(cols, key_indices, nparts, outputs)
        # Re-construct output partitions
        outdf = DataFrame()
        for k, col in zip(self._cols, outputs):
            outdf[k] = col
        # Slice into partition
        return [outdf[s:e] for s, e in zip(offsets, offsets[1:] + [None])]

    def replace(self, to_replace, value):
        """
        Replace values given in *to_replace* with *value*.

        Parameters
        ----------
        to_replace : numeric, str, list-like or dict
            Value(s) to replace.

            * numeric or str:

                - values equal to *to_replace* will be replaced
                  with *value*

            * list of numeric or str:

                - If *value* is also list-like,
                  *to_replace* and *value* must be of same length.

            * dict:

                - Dicts can be used to replace different values in different
                  columns. For example, `{'a': 1, 'z': 2}` specifies that the
                  value 1 in column `a` and the value 2 in column `z` should be
                  replaced with value*.
        value : numeric, str, list-like, or dict
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
        if not is_dict_like(value):
            value = dict.fromkeys(self.columns, value)

        for k in to_replace:
            outdf[k] = self[k].replace(to_replace[k], value[k])

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
            outdf[k] = self[k].fillna(value[k], method=method, axis=axis,
                                      inplace=inplace, limit=limit)

        if not inplace:
            return outdf

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
        index = self.index.to_pandas()
        out = pd.DataFrame(index=index)
        for c, x in self._cols.items():
            out[c] = x.to_pandas(index=index)
        return out

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
            raise TypeError('not a pandas.DataFrame')

        df = cls()
        # Set columns
        for colk in dataframe.columns:
            vals = dataframe[colk].values
            df[colk] = Series(vals, nan_as_null=nan_as_null)
        # Set index
        return df.set_index(dataframe.index)

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

        for name, column in self._cols.items():
            names.append(name)
            arrow_col = column.to_arrow()
            arrays.append(arrow_col)
            types.append(arrow_col.type)

        index_name = pa.pandas_compat._index_level_name(self.index, 0, names)
        index_names.append(index_name)
        index_columns.append(self.index)
        # It would be better if we didn't convert this if we didn't have to,
        # but we first need better tooling for cudf --> pyarrow type
        # conversions
        index_arrow = self.index.to_arrow()
        types.append(index_arrow.type)
        if preserve_index:
            arrays.append(index_arrow)
            names.append(index_name)

        # We may want to add additional metadata to this in the future, but
        # for now lets just piggyback off of what's done for Pandas
        metadata = pa.pandas_compat.construct_metadata(
            self, names, index_columns, index_names, preserve_index, types
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
        import json
        if not isinstance(table, pa.Table):
            raise TypeError('not a pyarrow.Table')

        index_col = None
        dtypes = None
        if isinstance(table.schema.metadata, dict):
            if b'pandas' in table.schema.metadata:
                metadata = json.loads(
                    table.schema.metadata[b'pandas']
                )
                index_col = metadata['index_columns']
                dtypes = {col['field_name']: col['pandas_type'] for col in
                          metadata['columns'] if 'field_name' in col}

        df = cls()
        for col in table.columns:
            if dtypes:
                dtype = dtypes[col.name]
                if dtype == 'categorical':
                    dtype = 'category'
                elif dtype == 'date':
                    dtype = 'datetime64[ms]'
            else:
                dtype = None

            df[col.name] = columnops.as_column(
                col.data,
                dtype=dtype
            )
        if index_col:
            df = df.set_index(index_col[0])
            new_index_name = pa.pandas_compat._backwards_compatible_index_name(
                df.index.name, df.index.name)
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
        members = [('index', self.index.dtype)] if index else []
        members += [(col, self[col].dtype) for col in self.columns]
        dtype = np.dtype(members)
        ret = np.recarray(len(self), dtype=dtype)
        if index:
            ret['index'] = self.index.values
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
            raise ValueError("records dimension expected 1 or 2 but found {!r}"
                             .format(data.ndim))

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
    def from_gpu_matrix(self, data, index=None, columns=None,
                        nan_as_null=False):
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
            raise ValueError("matrix dimension expected 2 but found {!r}"
                             .format(data.ndim))

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
    def quantile(self,
                 q=0.5,
                 interpolation='linear',
                 columns=None,
                 exact=True):
        """
        Return values at the given quantile.

        Parameters
        ----------

        q : float or array-like
            0 <= q <= 1, the quantile(s) to compute
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
        if columns is None:
            columns = self.columns

        result = DataFrame()
        result['Quantile'] = q
        for k, col in self._cols.items():
            if k in columns:
                result[k] = col.quantile(q, interpolation=interpolation,
                                         exact=exact,
                                         quant_index=False)
        return result

    def select_dtypes(self, include=None):
        """Return a subset of the DataFrame’s columns based on the column dtypes.

        Parameters
        ----------
        include : str or list
            which columns to include based on dtypes

        """

        if not isinstance(include, (list, tuple)):
            include = [include]
        df = DataFrame()

        include = [pd.core.dtypes.common.pandas_dtype(d) for d in include]

        for x in self._cols.values():
            try:
                if x.dtype in include:
                    df.add_column(x.name, x)
            except TypeError:
                pass
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
        json.to_json(
            self,
            path_or_buf=path_or_buf,
            *args,
            **kwargs
        )

    @ioutils.doc_to_hdf()
    def to_hdf(self, path_or_buf, key, *args, **kwargs):
        """{docstring}"""
        import cudf.io.hdf as hdf
        hdf.to_hdf(path_or_buf, key, self, *args, **kwargs)


class Loc(object):
    """
    For selection by label.
    """

    def __init__(self, df):
        self._df = df

    def __getitem__(self, arg):
        row_slice = None
        row_label = None

        if isinstance(arg, int):
            if arg < 0 or arg >= len(self._df):
                raise IndexError("label scalar %s is out of bound" % arg)
            row_label = arg
            col_slice = self._df.columns

        elif isinstance(arg, tuple):
            arg_1, arg_2 = arg
            if isinstance(arg_1, int):
                row_label = arg_1
            elif isinstance(arg_1, slice):
                row_slice = arg_1
            else:
                raise TypeError(type(arg_1))
            col_slice = arg_2

        elif isinstance(arg, slice):
            row_slice = arg
            col_slice = self._df.columns
        else:
            raise TypeError(type(arg))

        if row_label is not None:
            ret_list = []
            col_list = pd.Categorical(list(col_slice))
            for col in col_list:
                if pd.api.types.is_categorical_dtype(
                        self._df[col][row_label].dtype
                ):
                    raise NotImplementedError(
                        "categorical dtypes are not yet supported in loc"
                    )
                ret_list.append(self._df[col][row_label])
            promoted_type = np.result_type(*[val.dtype for val in ret_list])
            ret_list = np.array(ret_list, dtype=promoted_type)
            return Series(ret_list,
                          index=as_index(col_list))

        df = DataFrame()
        begin, end = self._df.index.find_label_range(row_slice.start,
                                                     row_slice.stop)
        row_step = row_slice.step if row_slice.step is not None else 1
        for col in col_slice:
            sr = self._df[col]
            df.add_column(col, sr[begin:end:row_step], forceindex=True)

        return df


class Iloc(object):
    """
    For integer-location based selection.
    """

    def __init__(self, df):
        self._df = df

    def __getitem__(self, arg):
        rows = []
        len_idx = len(self._df.index)

        if isinstance(arg, tuple):
            raise NotImplementedError('cudf columnar iloc not supported')

        elif isinstance(arg, int):
            rows.append(arg)

        elif isinstance(arg, slice):
            start, stop, step, sln = utils.standard_python_slice(len_idx, arg)
            if sln > 0:
                for idx in range(start, stop, step):
                    rows.append(idx)

        elif isinstance(arg, utils.list_types_tuple):
            for idx in arg:
                rows.append(idx)

        else:
            raise TypeError(type(arg))

        # To check whether all the indices are valid.
        for idx in rows:
            if abs(idx) > len_idx or idx == len_idx:
                raise IndexError("positional indexers are out-of-bounds")

        # returns the series similar to pandas
        if isinstance(arg, int) and len(rows) == 1:
            ret_list = []
            col_list = pd.Categorical(list(self._df.columns))
            for col in col_list:
                if pd.api.types.is_categorical_dtype(
                    self._df[col][rows[0]].dtype
                ):
                    raise NotImplementedError(
                        "categorical dtypes are not yet supported in iloc"
                    )
                ret_list.append(self._df[col][rows[0]])
            promoted_type = np.result_type(*[val.dtype for val in ret_list])
            ret_list = np.array(ret_list, dtype=promoted_type)
            return Series(ret_list,
                          index=as_index(col_list))

        df = DataFrame()

        for col in self._df.columns:
            sr = self._df[col]
            df.add_column(col, sr.iloc[tuple(rows)])

        # 0-length rows can occur when when iloc[n=0]
        # head(0)
        if isinstance(arg, slice):
            df.index = sr.index[arg]
        else:
            df.index = sr.index[rows]
        return df

    def __setitem__(self, key, value):
        # throws an exception while updating
        msg = "updating columns using iloc is not allowed"
        raise ValueError(msg)


def from_pandas(obj):
    """
    Convert a Pandas DataFrame or Series object into the cudf equivalent

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
    else:
        raise TypeError(
            "from_pandas only accepts Pandas Dataframes and Series objects. "
            "Got %s" % type(obj)
        )


def merge(left, right, *args, **kwargs):
    return left.merge(right, *args, **kwargs)


# a bit of fanciness to inject doctstring with left parameter
merge_doc = DataFrame.merge.__doc__
idx = merge_doc.find('right')
merge.__doc__ = ''.join([merge_doc[:idx], '\n\tleft : DataFrame\n\t',
                        merge_doc[idx:]])

register_distributed_serializer(DataFrame)
