from __future__ import print_function, division

import inspect
from collections import OrderedDict

import numpy as np
import pandas as pd

from numba import cuda

from . import cudautils, formatting, queryutils
from .index import Int64Index, EmptyIndex
from .series import Series


class DataFrame(object):
    """
    A GPU Dataframe object.

    Examples
    --------

    Build dataframe with `__setitem__`

    >>> from pygdf.dataframe import DataFrame
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
    >>> from pygdf.dataframe import DataFrame
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

    def __init__(self, name_series=None):
        self._index = EmptyIndex()
        self._size = 0
        self._cols = OrderedDict()
        # has initializer?
        if name_series is not None:
            for k, series in name_series:
                self.add_column(k, series)

    def __getitem__(self, arg):
        """
        If *arg* is a ``str``, return the column Series.
        If *arg* is a ``slice``, return a new DataFrame with all columns
        sliced to the specified range.

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
        """
        if isinstance(arg, str):
            return self._cols[arg]
        elif isinstance(arg, slice):
            df = DataFrame()
            for k, col in self._cols.items():
                df[k] = col[arg]
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

    def __len__(self):
        """Returns the number of rows
        """
        return self._size

    def to_string(self, nrows=5, ncols=8):
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
        if nrows is None:
            nrows = len(self)
        else:
            nrows = min(nrows, len(self))  # cap row count

        if ncols is None:
            ncols = len(self)

        more_cols = len(self.columns) - ncols
        more_rows = len(self) - nrows

        # Prepare cells
        cols = OrderedDict()
        for h in self.columns[:ncols]:
            cols[h] = self[h].values_to_string(nrows=nrows)
        # Format into a table
        return formatting.format(index=self._index, cols=cols,
                                 show_headers=True, more_cols=more_cols,
                                 more_rows=more_rows)

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

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
        return tuple(self._cols)

    @property
    def index(self):
        """Returns the index of the DataFrame
        """
        return self._index

    def set_index(self, index):
        """Return a new DataFrame with a new index
        """
        df = self.copy()
        df._index = index
        return df

    def copy(self):
        "Shallow copy this dataframe"
        df = DataFrame()
        for k in self.columns:
            df[k] = self[k]
        return df

    def _prepare_series_for_add(self, col):
        """Prepare a series to be added to the DataFrame.

        Parameters
        ----------
        col : Series, array-like
            Values to be added.

        Returns
        -------
        The prepared Series object.
        """
        series = Series.from_any(col)
        empty_index = isinstance(self._index, EmptyIndex)
        if empty_index or self._index == series.index:
            if empty_index:
                self._index = series.index
            self._size = len(series)
            return series
        else:
            raise NotImplementedError("join needed")

    def add_column(self, name, data):
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
        series = self._prepare_series_for_add(data)
        self._cols[name] = series

    def drop_column(self, name):
        """Drop a column by *name*
        """
        if name not in self._cols:
            raise NameError('column {!r} does not exist'.format(name))
        del self._cols[name]

    def concat(self, *dfs):
        """Concat rows from other dataframes.

        Parameters
        ----------

        *dfs : one or more DataFrame(s)

        Returns
        -------

        A new dataframe with rows from each dataframe in ``*dfs``.
        """
        # check columns
        for df in dfs:
            if df.columns != self.columns:
                raise ValueError('columns mismatch')

        newdf = DataFrame()
        # foreach column
        for k, col in self._cols.items():
            # append new rows to the column
            for df in dfs:
                col = col.append(df[k])
            newdf[k] = col
        return newdf

    def as_gpu_matrix(self, columns=None):
        """Covert to a matrix in device memory.

        Parameters
        ----------
        columns: sequence of str
            List of a column names to be extracted.  The order is preserved.
            If None is specified, all columns are used.

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
            raise ValueError('all column must have the same dtype')
        for k, c in self._cols.items():
            if c.has_null_mask:
                raise ValueError("column {!r} is sparse".format(k))

        matrix = cuda.device_array(shape=(nrow, ncol), dtype=dtype, order="F")
        for colidx, inpcol in enumerate(cols):
            dense = inpcol.to_gpu_array(fillna='pandas')
            matrix[:, colidx].copy_to_device(dense)

        return matrix

    def as_matrix(self, columns=None):
        """Covert to a matrix in host memory.

        Parameters
        ----------
        columns: sequence of str
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
        """
        newnames = [prefix_sep.join([prefix, str(cat)]) for cat in cats]
        newcols = self[column].one_hot_encoding(cats=cats, dtype=dtype)
        outdf = self.copy()
        for name, col in zip(newnames, newcols):
            outdf.add_column(name, col)
        return outdf

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
        sorted_indices = self[by].argsort()
        index = Int64Index(sorted_indices.to_gpu_array())
        df = DataFrame()
        # Perform out = data[index] for all columns
        for k in self.columns:
            col = self[k]
            out = cudautils.gather(data=col.to_gpu_array(),
                                   index=sorted_indices.to_gpu_array())
            sr = Series.from_array(out).set_index(index)
            df[k] = sr
        return df

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
        col = self[column]
        # Operate
        sorted_series = getattr(col, method)(n=n, keep=keep)
        df = DataFrame()
        for k in self.columns:
            if k == column:
                df[k] = sorted_series
            else:
                df[k] = self[k].take(df.index.gpu_values)
        return df

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
        # Get calling environment
        callframe = inspect.currentframe().f_back
        callenv = {
            'locals': callframe.f_locals,
            'globals': callframe.f_globals,
        }
        # Run query
        boolmask = queryutils.query_execute(self, expr, callenv)

        selected = Series.from_array(boolmask)
        newdf = DataFrame()
        for col in self.columns:
            newseries = self[col][selected]
            newdf[col] = newseries
        return newdf

    def to_pandas(self):
        """Convert to a Pandas DataFrame.
        """
        dct = {k: c.to_array(fillna='pandas') for k, c in self._cols.items()}
        return pd.DataFrame.from_dict(dct)

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

        for colk in dataframe.columns:
            df[colk] = dataframe[colk].values
        return df

    def to_records(self, index=True):
        """Covert to a numpy recarray

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
        """Convert from a numpy recarray or structured array

        Parameters
        ----------
        data : numpy structured dtype or recarray
        index : str
            The name of the index column in *data*.
            If None, the default index is used.
        columns: list of str
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
            return df.set_index(Int64Index(indices.astype(np.int64)))
        return df


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
        for col in col_slice:
            sr = self._df[col]
            begin, end = sr.index.find_label_range(row_slice.start,
                                                   row_slice.stop)
            df[col] = sr[begin:end]
        return df


