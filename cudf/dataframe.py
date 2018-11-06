# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import print_function, division

import inspect
import random
from collections import OrderedDict
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa

from numba.cuda.cudadrv.devicearray import DeviceNDArray

from librmm_cffi import librmm as rmm

from . import cudautils, formatting, queryutils, applyutils, utils, _gdf
from .index import GenericIndex, Index, RangeIndex
from .series import Series
from .column import Column
from .settings import NOTSET, settings
from .serialize import register_distributed_serializer
from .categorical import CategoricalColumn
from .datetime import DatetimeColumn
from .numerical import NumericalColumn
from .buffer import Buffer
from ._gdf import nvtx_range_push, nvtx_range_pop


class DataFrame(object):
    """
    A GPU Dataframe object.

    Examples
    --------

    Build dataframe with `__setitem__`

    >>> from cudf.dataframe import DataFrame
    >>> df = DataFrame()
    >>> df['key'] = [0, 1, 2, 3, 4]
    >>> df['val'] = [float(i + 10) for i in range(5)]  # insert column
    >>> df
      key val
    0 0   10.0
    1 1   11.0
    2 2   12.0
    3 3   13.0
    4 4   14.0
    >>> len(df)
    5

    Build dataframe with initializer

    >>> import numpy as np
    >>> df2 = DataFrame([('a', np.arange(10)),
    ...                  ('b', np.random.random(10))])
    >>> df2
      a b
    0 0 0.777831724018
    1 1 0.604480034669
    2 2 0.664111858618
    3 3 0.887777513028
    4 4 0.55838311246
    [5 more rows]

    Convert from a Pandas DataFrame.

    >>> import pandas as pd
    >>> from cudf.dataframe import DataFrame
    >>> pdf = pd.DataFrame({'a': [0, 1, 2, 3],
    ...                     'b': [0.1, 0.2, None, 0.3]})
    >>> pdf
    a    b
    0  0  0.1
    1  1  0.2
    2  2  NaN
    3  3  0.3
    >>> df = DataFrame.from_pandas(pdf)
    >>> df
    a b
    0 0 0.1
    1 1 0.2
    2 2 nan
    3 3 0.3
    """

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
        If *arg* is a ``str``, return the column Series.
        If *arg* is a ``slice``, return a new DataFrame with all columns
        sliced to the specified range.
        If *arg* is an ``array`` containing column names, return a new
        DataFrame with the corresponding columns.


        Examples
        --------
        >>> df = DataFrame([('a', list(range(20))),
        ...                 ('b', list(range(20))),
        ...                 ('c', list(range(20)))])
        >>> df[:4]    # get first 4 rows of all columns
             a    b    c
        0    0    0    0
        1    1    1    1
        2    2    2    2
        3    3    3    3
        >>> df[-5:]  # get last 5 rows of all columns
             a    b    c
        15   15   15   15
        16   16   16   16
        17   17   17   17
        18   18   18   18
        19   19   19   19
        >>>df[['a','c']] # get columns a and c
             a    c
        0    0    0
        1    1    1
        2    2    2
        3    3    3
        """
        if isinstance(arg, str) or isinstance(arg, int):
            s = self._cols[arg]
            s.name = arg
            return s
        elif isinstance(arg, slice):
            df = DataFrame()
            for k, col in self._cols.items():
                df[k] = col[arg]
            return df
        elif isinstance(arg, (list,)):
            df = DataFrame()
            for col in arg:
                df[col] = self[col]
            return df
        else:
            msg = "__getitem__ on type {!r} is not supported"
            raise TypeError(msg.format(arg))

    def __setitem__(self, name, col):
        """Add/set column by *name*
        """

        if name in self._cols:
            self._cols[name] = self._prepare_series_for_add(col)
        else:
            self.add_column(name, col)

    def __delitem__(self, name):
        """Drop the give column by *name*.
        """
        self.drop_column(name)

    def __sizeof__(self):
        return sum(col.__sizeof__() for col in self._cols.values())

    def __len__(self):
        """Returns the number of rows
        """
        return self._size

    def assign(self, **kwargs):
        new = self.copy()
        for k, v in kwargs.items():
            new[k] = v
        return new

    def head(self, n=5):
        return self[:n]

    def to_string(self, nrows=NOTSET, ncols=NOTSET):
        """Convert to string

        Parameters
        ----------
        nrows : int
            Maximum number of rows to show.
            If it is None, all rows are shown.

        ncols : int
            Maximum number of columns to show.
            If it is None, all columns are shown.
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
                                 more_rows=more_rows)

    def __str__(self):
        nrows = settings.formatting.get('nrows') or 10
        ncols = settings.formatting.get('ncols') or 8
        return self.to_string(nrows=nrows, ncols=ncols)

    def __repr__(self):
        return "<cudf.DataFrame ncols={} nrows={} >".format(
            len(self.columns),
            len(self),
        )

    @property
    def loc(self):
        """
        Returns a label-based indexer for row-slicing and column selection.

        Examples
        --------

        >>> df = DataFrame([('a', list(range(20))),
        ...                 ('b', list(range(20))),
        ...                 ('c', list(range(20)))])
        # get rows from index 2 to index 5 from 'a' and 'b' columns.
        >>> df.loc[2:5, ['a', 'b']]
             a    b
        2    2    2
        3    3    3
        4    4    4
        5    5    5
        """
        return Loc(self)

    @property
    def columns(self):
        """Returns a tuple of columns
        """
        return pd.Index(self._cols)

    @property
    def index(self):
        """Returns the index of the DataFrame
        """
        return self._index

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
            df = self.copy()
            df.drop_column(index)
            return df.set_index(self[index])
        # Otherwise
        else:
            index = index if isinstance(index, Index) else GenericIndex(index)
            df = DataFrame()
            for k in self.columns:
                df[k] = self[k].set_index(index)
            return df

    def reset_index(self):
        return self.set_index(RangeIndex(len(self)))

    def take(self, positions, ignore_index=False):
        out = DataFrame()
        for col in self.columns:
            out[col] = self[col].take(positions, ignore_index=ignore_index)
        return out

    def copy(self):
        """
        copy this dataframe
        """
        df = DataFrame()
        df._index = self._index
        df._size = self._size
        df._cols = self._cols.copy()
        return df

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo={}):
        """
        Parameters
        ----------
        memo, default None
            Standard signature. Unused
        """
        if memo is None:
            memo = {}
        return self.copy()

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
        VALID = isinstance(col, (np.ndarray, DeviceNDArray, list, Series,
                                 Column))
        if len(self) > 0 and len(series) == 1 and not VALID:
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
        if forceindex or empty_index or self._index == series.index:
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

        series = self._prepare_series_for_add(data, forceindex=forceindex)
        series.name = name
        self._cols[name] = series

    def drop_column(self, name):
        """Drop a column by *name*
        """
        if name not in self._cols:
            raise NameError('column {!r} does not exist'.format(name))
        del self._cols[name]

    @classmethod
    def _concat(cls, objs, ignore_index=False):
        nvtx_range_push("PYGDF_CONCAT", "orange")
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
        """Expand a column with one-hot-encoding.

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
        -------
        >>> import pandas as pd
        >>> from cudf.dataframe import DataFrame as gdf

        >>> pet_owner = [1, 2, 3, 4, 5]
        >>> pet_type = ['fish', 'dog', 'fish', 'bird', 'fish']

        >>> df = pd.DataFrame({'pet_owner': pet_owner, 'pet_type': pet_type})
        >>> df.pet_type = df.pet_type.astype('category')

        Create a column with numerically encoded category values
        >>> df['pet_codes'] = df.pet_type.cat.codes
        >>> my_gdf = gdf.from_pandas(df)

        Create the list of category codes to use in the encoding
        >>> codes = my_gdf.pet_codes.unique()
        >>> enc_gdf = my_gdf.one_hot_encoding('pet_codes', 'pet_dummy', codes)
        >>> enc_gdf.head()
          pet_owner pet_type pet_codes pet_dummy_0 pet_dummy_1 pet_dummy_2
          0         1     fish         2         0.0         0.0         1.0
          1         2      dog         1         0.0         1.0         0.0
          2         3     fish         2         0.0         0.0         1.0
          3         4     bird         0         1.0         0.0         0.0
          4         5     fish         2         0.0         0.0         1.0
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

    def sort_index(self, ascending=True):
        """Sort by the index
        """
        return self._sort_by(self.index.argsort(ascending=ascending))

    def sort_values(self, by, ascending=True):
        """
        Sort by values.

        Difference from pandas:
        * *by* must be the name of a single column.
        * Support axis='index' only.
        * Not supporting: inplace, kind, na_position

        Details:
        Uses parallel radixsort, which is a stable sort.
        """
        # argsort the `by` column
        return self._sort_by(self[by].argsort(ascending=ascending))

    def nlargest(self, n, columns, keep='first'):
        """Get the rows of the DataFrame sorted by the n largest value of *columns*

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
        col = self[column].reset_index()
        # Operate
        sorted_series = getattr(col, method)(n=n, keep=keep)
        df = DataFrame()
        new_positions = sorted_series.index.gpu_values
        for k in self.columns:
            if k == column:
                df[k] = sorted_series
            else:
                df[k] = self[k].reset_index().take(new_positions)
        return df.set_index(self.index.take(new_positions))

    def merge(self, other, on=None, how='left', lsuffix='_x', rsuffix='_y',
              type="", method='hash'):
        """Merge GPU DataFrame objects by performing a database-style join operation
        by columns or indexes.

        Parameters
        ----------
        other : DataFrame
        on : label or list; defaults to None
            Column or index level names to join on. These must be found in
            both DataFrames.

            If on is None and not merging on indexes then
            this defaults to the intersection of the columns
            in both DataFrames.
        how : str, defaults to 'left'
            Only accepts 'left'
            left: use only keys from left frame, similar to
            a SQL left outer join; preserve key order
        lsuffix : str, defaults to '_x'
            Suffix applied to overlapping column names on the left side
        rsuffix : str, defaults to '_y'
            Suffix applied to overlapping column names on the right side
        type : str, defaults to 'hash'

        Returns
        -------
        merged : DataFrame

        """
        _gdf.nvtx_range_push("PYGDF_JOIN", "blue")

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

        same_names = set(self.columns) & set(other.columns)
        if same_names and not (lsuffix or rsuffix):
            raise ValueError('there are overlapping columns but '
                             'lsuffix and rsuffix are not defined')

        def fix_name(name, suffix):
            if name in same_names:
                return "{}{}".format(name, suffix)
            return name

        lhs = self
        rhs = other

        col_cats = {}

        for name in on:
            if pd.api.types.is_categorical_dtype(self[name]):
                lcats = self[name].cat.categories
                rcats = other[name].cat.categories
                if how == 'left':
                    cats = lcats
                    other[name] = (other[name].cat.set_categories(cats)
                                   .fillna(-1))
                elif how == 'right':
                    cats = rcats
                    self[name] = (self[name].cat.set_categories(cats)
                                  .fillna(-1))
                elif how in ['inner', 'outer']:
                    # Do the join using the union of categories from both side.
                    # Adjust for inner joins afterwards
                    cats = sorted(set(lcats) | set(rcats))

                    self[name] = (self[name].cat.set_categories(cats)
                                  .fillna(-1))
                    self[name] = self[name]._column.as_numerical

                    other[name] = (other[name].cat.set_categories(cats)
                                   .fillna(-1))
                    other[name] = other[name]._column.as_numerical

                col_cats[name] = cats

        for name, col in lhs._cols.items():
            if pd.api.types.is_categorical_dtype(col) and name not in on:
                f_n = fix_name(name, lsuffix)
                col_cats[f_n] = self[name].cat.categories

        for name, col in rhs._cols.items():
            if pd.api.types.is_categorical_dtype(col) and name not in on:
                f_n = fix_name(name, rsuffix)
                col_cats[f_n] = other[name].cat.categories

        cols, valids = _gdf.libgdf_join(lhs._cols, rhs._cols, on, how,
                                        method=method)

        df = DataFrame()

        # Columns are returned in order left - on - right from libgdf
        # Creating dataframe with ordering as pandas:

        gap = len(self.columns) - len(on)
        for idx in range(len(on)):
            if (cols[idx + gap].dtype == 'datetime64[ms]'):
                df[on[idx]] = DatetimeColumn(data=Buffer(cols[idx + gap]),
                                             dtype=np.dtype('datetime64[ms]'),
                                             mask=Buffer(valids[idx]))
            elif on[idx] in col_cats.keys():
                df[on[idx]] = CategoricalColumn(data=Buffer(cols[idx + gap]),
                                                categories=col_cats[on[idx]],
                                                dtype='category',
                                                ordered=False,
                                                mask=Buffer(valids[idx]))
            else:
                df[on[idx]] = NumericalColumn(data=Buffer(cols[idx + gap]),
                                              dtype=cols[idx + gap].dtype,
                                              mask=Buffer(valids[idx]))

        idx = 0

        for name in self.columns:
            if name not in on:
                f_n = fix_name(name, lsuffix)
                if (cols[idx].dtype == 'datetime64[ms]'):
                    df[f_n] = DatetimeColumn(data=Buffer(cols[idx]),
                                             dtype=np.dtype('datetime64[ms]'),
                                             mask=Buffer(valids[idx]))
                elif f_n in col_cats.keys():
                    df[f_n] = CategoricalColumn(data=Buffer(cols[idx]),
                                                categories=col_cats[f_n],
                                                dtype='category',
                                                ordered=False,
                                                mask=Buffer(valids[idx]))
                else:
                    df[f_n] = NumericalColumn(data=Buffer(cols[idx]),
                                              dtype=cols[idx].dtype,
                                              mask=Buffer(valids[idx]))
                idx = idx + 1

        idx = len(self.columns)

        for name in other.columns:
            if name not in on:
                f_n = fix_name(name, rsuffix)
                if (cols[idx].dtype == 'datetime64[ms]'):
                    df[f_n] = DatetimeColumn(data=Buffer(cols[idx]),
                                             dtype=np.dtype('datetime64[ms]'),
                                             mask=Buffer(valids[idx]))
                elif f_n in col_cats.keys():
                    df[f_n] = CategoricalColumn(data=Buffer(cols[idx]),
                                                categories=col_cats[f_n],
                                                dtype='categorical',
                                                ordered=False,
                                                mask=Buffer(valids[idx]))
                else:
                    df[f_n] = NumericalColumn(data=Buffer(cols[idx]),
                                              dtype=cols[idx].dtype,
                                              mask=Buffer(valids[idx]))
                idx = idx + 1

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

        _gdf.nvtx_range_push("PYGDF_JOIN", "blue")

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

        lhs = lhs.reset_index()
        rhs = rhs.reset_index()

        cat_join = False

        if pd.api.types.is_categorical_dtype(lhs[idx_col_name]):
            cat_join = True
            lcats = lhs[idx_col_name].cat.categories
            rcats = rhs[idx_col_name].cat.categories
            if how == 'left':
                cats = lcats
                rhs[idx_col_name] = (rhs[idx_col_name].cat
                                                      .set_categories(cats)
                                                      .fillna(-1))
            elif how == 'right':
                cats = rcats
                lhs[idx_col_name] = (lhs[idx_col_name].cat
                                                      .set_categories(cats)
                                                      .fillna(-1))
            elif how in ['inner', 'outer']:
                cats = sorted(set(lcats) | set(rcats))

                lhs[idx_col_name] = (lhs[idx_col_name].cat
                                                      .set_categories(cats)
                                                      .fillna(-1))
                lhs[idx_col_name] = lhs[idx_col_name]._column.as_numerical

                rhs[idx_col_name] = (rhs[idx_col_name].cat
                                                      .set_categories(cats)
                                                      .fillna(-1))
                rhs[idx_col_name] = rhs[idx_col_name]._column.as_numerical

                print(cats)
                print(lhs[idx_col_name])
                print(rhs[idx_col_name])

        if lsuffix == '':
            lsuffix = 'l'
        if rsuffix == '':
            rsuffix = 'r'

        df = lhs.merge(rhs, on=[idx_col_name], how=how, lsuffix=lsuffix,
                       rsuffix=rsuffix, method=method)

        if cat_join:
            df[idx_col_name] = CategoricalColumn(data=df[idx_col_name].data,
                                                 categories=cats,
                                                 dtype='categorical',
                                                 ordered=False)

        df = df.set_index(idx_col_name)

        if sort and len(df):
            return df.sort_index()

        return df

    def groupby(self, by, sort=False, as_index=False, method="sort"):
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
            Valid values are "sort", "hash", or "cudf".
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
        if (method == "cudf"):
            from .groupby import Groupby
            if as_index:
                msg = "as_index==True not supported due to the lack of\
                    multi-index"
                raise NotImplementedError(msg)
            result = Groupby(self, by=by)
            return result
        else:
            from .libgdf_groupby import LibGdfGroupby

            _gdf.nvtx_range_push("PYGDF_GROUPBY", "purple")
            if as_index:
                msg = "as_index==True not supported due to the lack of\
                    multi-index"
                raise NotImplementedError(msg)
            # The matching `pop` for this range is inside LibGdfGroupby
            # __apply_agg
            result = LibGdfGroupby(self, by=by, method=method)
            return result

    def query(self, expr):
        """Query with a boolean expression using Numba to compile a GPU kernel.

        See pandas.DataFrame.query.

        Parameters
        ----------
        expr : str
            A boolean expression.  Names in the expression refers to the
            columns.  Any name prefixed with `@` refer to the variables in
            the calling environment.

        Returns
        -------
        filtered :  DataFrame
        """

        _gdf.nvtx_range_push("PYGDF_QUERY", "purple")
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
        """Transform each row using the user-provided function.

        Parameters
        ----------
        {params}

        Examples
        --------

        With a ``DataFrame`` like so:

        >>> df = DataFrame()
        >>> df['in1'] = in1 = np.arange(nelem)
        >>> df['in2'] = in2 = np.arange(nelem)
        >>> df['in3'] = in3 = np.arange(nelem)

        Define the user function for ``.apply_rows``:

        >>> def kernel(in1, in2, in3, out1, out2, extra1, extra2):
        ...     for i, (x, y, z) in enumerate(zip(in1, in2, in3)):
        ...         out1[i] = extra2 * x - extra1 * y
        ...         out2[i] = y - extra1 * z

        The user function should loop over the columns and set the output for
        each row.  Each iteration of the loop **MUST** be independent of each
        other.  The order of the loop execution can be arbitrary.

        Call ``.apply_rows`` with the name of the input columns, the name and
        dtype of the output columns, and, optionally, a dict of extra
        arguments.

        >>> outdf = df.apply_rows(kernel,
        ...                       incols=['in1', 'in2', 'in3'],
        ...                       outcols=dict(out1=np.float64,
        ...                                    out2=np.float64),
        ...                       kwargs=dict(extra1=2.3, extra2=3.4))

        **Notes**

        When ``func`` is invoked, the array args corresponding to the
        input/output are strided in a way that improves parallelism on the GPU.
        The loop in the function may look like serial code but it will be
        executed concurrently by multiple threads.
        """
        return applyutils.apply_rows(self, func, incols, outcols, kwargs,
                                     cache_key=cache_key)

    @applyutils.doc_applychunks()
    def apply_chunks(self, func, incols, outcols, kwargs={}, chunks=None,
                     tpb=1):
        """Transform user-specified chunks using the user-provided function.

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
        specified chunk.  The *kernel* computes the corresponding output
        for the chunk.  By looping over the range
        ``range(cuda.threadIdx.x, in1.size, cuda.blockDim.x)``, the *kernel*
        function can be used with any *tpb* in a efficient manner.

        >>> from numba import cuda
        >>> def kernel(in1, in2, in3, out1):
        ...     for i in range(cuda.threadIdx.x, in1.size, cuda.blockDim.x):
        ...         x = in1[i]
        ...         y = in2[i]
        ...         z = in3[i]
        ...         out1[i] = x * y + z

        See also
        --------
        .apply_rows

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
        from . import numerical

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

    def to_pandas(self):
        """Convert to a Pandas DataFrame.
        """
        index = self.index.to_pandas()
        data = {c: x.to_pandas(index=index) for c, x in self._cols.items()}
        return pd.DataFrame(data, columns=list(self._cols), index=index)

    @classmethod
    def from_pandas(cls, dataframe):
        """Convert from a Pandas DataFrame.

        Raises
        ------
        TypeError for invalid input type.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError('not a pandas.DataFrame')

        df = cls()
        # Set columns
        for colk in dataframe.columns:
            df[colk] = dataframe[colk].values
        # Set index
        return df.set_index(dataframe.index.values)

    def to_arrow(self, index=True):
        """Convert to a PyArrow Table.
        """
        arrays = []
        names = []
        if index:
            names.append(self.index.name)
            arrays.append(self.index.to_arrow())
        for name, column in self._cols.items():
            names.append(name)
            arrays.append(column.to_arrow())
        return pa.Table.from_arrays(arrays, names=names)

    @classmethod
    def from_arrow(cls, table):
        """Convert from a PyArrow Table.

        Raises
        ------
        TypeError for invalid input type.

        **Notes**

        Does not support automatically setting index column(s) similar to how
        ``to_pandas`` works for PyArrow Tables.
        """
        if not isinstance(table, pa.Table):
            raise TypeError('not a pyarrow.Table')

        df = cls()
        for col in table.columns:
            if len(col.data.chunks) != 1:
                raise NotImplementedError("Importing from PyArrow Tables "
                                          "with multiple chunks is not yet "
                                          "supported")
            df[col.name] = col.data.chunk(0)
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
    def from_records(self, data, index=None, columns=None):
        """Convert from a numpy recarray or structured array.

        Parameters
        ----------
        data : numpy structured dtype or recarray
        index : str
            The name of the index column in *data*.
            If None, the default index is used.
        columns : list of str
            List of column names to include.

        Returns
        -------
        DataFrame
        """
        names = data.dtype.names if columns is None else columns
        df = DataFrame()
        for k in names:
            # FIXME: unnecessary copy
            df[k] = np.ascontiguousarray(data[k])
        if index is not None:
            indices = data[index]
            return df.set_index(indices.astype(np.int64))
        return df

    def quantile(self, q, interpolation='linear', exact=False):
        """Return values at the given quantile.
        Parameters
        ----------
        q : float or array-like
            0 <= q <= 1, the quantile(s) to compute
        interpolation : {‘linear’, ‘lower’, ‘higher’, ‘midpoint’, ‘nearest’}
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
        result = DataFrame()
        result['Quantile'] = q
        for k, col in self._cols.items():
            result[k] = col.quantile(q, interpolation, exact,
                                     quant_index=False)
        print(result)


class Loc(object):
    """
    For selection by label.
    """

    def __init__(self, df):
        self._df = df

    def __getitem__(self, arg):
        if isinstance(arg, tuple):
            row_slice, col_slice = arg
        elif isinstance(arg, slice):
            row_slice = arg
            col_slice = self._df.columns
        else:
            raise TypeError(type(arg))

        df = DataFrame()
        begin, end = self._df.index.find_label_range(row_slice.start,
                                                     row_slice.stop)
        for col in col_slice:
            sr = self._df[col]
            df.add_column(col, sr[begin:end], forceindex=True)

        return df


register_distributed_serializer(DataFrame)
