# Copyright (c) 2018, NVIDIA CORPORATION.

from __future__ import print_function, division

import inspect
from collections import OrderedDict

import numpy as np
import pandas as pd

from numba import cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from . import cudautils, formatting, queryutils, applyutils, utils, _gdf
from .index import GenericIndex, Index, RangeIndex
from .buffer import Buffer
from .series import Series
from .column import Column
from .settings import NOTSET, settings
from .serialize import register_distributed_serializer


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

    def __init__(self, name_series=None, index=None):
        if index is None:
            index = RangeIndex(start=0)
        self._index = index
        self._size = len(index)
        self._cols = OrderedDict()
        # has initializer?
        if name_series is not None:
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
            return self._cols[arg]
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

        more_cols = len(self.columns) - ncols
        more_rows = len(self) - nrows

        # Prepare cells
        cols = OrderedDict()
        use_cols = list(self.columns[:ncols - 1])
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
        return "<pygdf.DataFrame ncols={} nrows={} >".format(
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
        return tuple(self._cols)

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
        "Shallow copy this dataframe"
        df = DataFrame()
        df._index = self._index
        df._size = self._size
        df._cols = self._cols.copy()
        return df

    def _sanitize_columns(self, col):
        """Sanitize pre-appended
           col values
        """
        series = Series(col)
        if len(self) == 0 and len(self.columns) > 0 and len(series) > 0:
            ind = series.index
            arr = cuda.device_array(shape=len(ind), dtype=np.float64)
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
            arr = cuda.device_array(shape=len(index), dtype=series.dtype)
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
            raise NotImplementedError("join needed")

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
        self._cols[name] = series

    def drop_column(self, name):
        """Drop a column by *name*
        """
        if name not in self._cols:
            raise NameError('column {!r} does not exist'.format(name))
        del self._cols[name]

    @classmethod
    def _concat(cls, objs, ignore_index=False):
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
            if c.has_null_mask:
                errmsg = ("column {!r} has null values"
                          "hint: use .fillna() to replace null values")
                raise ValueError(errmsg.format(k))

        if order == 'F':
            matrix = cuda.device_array(shape=(nrow, ncol), dtype=dtype,
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
        >>> from pygdf.dataframe import DataFrame as gdf

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
              method='hash'):
        """Merge a DataFrame object with another DataFrame by performing a
        database-style join operation on columns.

        Parameters
        ----------
        other : DataFrame
        how : str
            Only accepts "left", "right", "inner", "outer"
        lsuffix, rsuffix : str
            The suffices to add to the left (*lsuffix*) and right (*rsuffix*)
            column names when avoiding conflicts.
        method: str, optional
            A string indicating the method to use to perform the merge.
            Valid values are "sort" or "hash"

        Returns
        -------
        The merged dataframe
        """
        if how not in ['left', 'right', 'inner', 'outer']:
            raise NotImplementedError('unsupported {!r} join'.format(how))

        same_names = set(self.columns) & set(other.columns)
        if same_names and not (lsuffix or rsuffix):
            raise ValueError('there are overlapping columns but '
                             'lsuffix and rsuffix are not defined')

        if how == 'right':
            # libgdf doesn't support right join directly, we will swap the
            # dfs and use left join
            return other.merge(other=self, on=on, how='left', lsuffix=rsuffix,
                               rsuffix=lsuffix, method=method)

        lhs = self
        rhs = other
        # XXX: Replace this stub
        # joined_values, joined_indicies = self._stub_merge(
        #    lhs, rhs, left_on=on, right_on=on, how=how,
        #    return_joined_indicies=True)
        joined_values, joined_indicies = self._merge_gdf(
            lhs, rhs, left_on=on, right_on=on, how=how, method=method,
            return_indices=True)

        return df

    def _merge_gdf(self, left, right, left_on, right_on, how, method,
                   return_indices):

        from pygdf import cudautils

        assert return_indices
        assert len(left_on) == len(right_on)

        left_cols = []
        for l in left_on:
            left_cols.append(left[l]._column)

        right_cols = []
        for r in right_on:
            right_cols.append(right[r]._column)

        joined_indices = []
        with _gdf.apply_join(left_cols, right_cols, how, method) \
                as (left_indices, right_indices):
            if left_indices.size > 0:
                # For each column we joined on, gather the values from each
                # column using the indices from the join
                joined_values = []

                for i in range(len(left_on)):
                    # TODO Instead of calling 'gather_joined_index' for every
                    # column that we are joining on, we should implement a
                    # 'multi_gather_joined_index' that can gather a value from
                    # each column at once
                    raw_values = cudautils.gather_joined_index(
                        left_cols[i].to_gpu_array(),
                        right_cols[i].to_gpu_array(),
                        left_indices,
                        right_indices,
                    )
                    buffered_values = Buffer(raw_values)

                    joined_values.append(left_cols[i]
                                         .replace(data=buffered_values))

                joined_indices = (cudautils.copy_array(left_indices),
                                  cudautils.copy_array(right_indices))

        # XXX: Prepare output.  same as _join.  code duplication
        def fix_name(name, suffix):
            if name in same_names:
                return "{}{}".format(name, suffix)
            return name

        def gather_cols(outdf, indf, on, idx, joinidx, suffix):
            mask = (Series(idx) != -1).as_mask()
            for k in on:
                newcol = indf[k].take(idx).set_mask(mask).set_index(joinidx)
                outdf[fix_name(k, suffix)] = newcol

        def gather_empty(outdf, indf, idx, joinidx, suffix):
            for k in indf.columns:
                outdf[fix_name(k, suffix)] = indf[k][:0]

        df = DataFrame()
        for key, col in zip(on, joined_values):
            df[key] = col

        left_indices, right_indices = joined_indicies
        gather_cols(df, lhs, [x for x in lhs.columns if x not in on],
                    left_indices, df.index, lsuffix)
        gather_cols(df, rhs, [x for x in rhs.columns if x not in on],
                    right_indices, df.index, rsuffix)

        if return_indices:
            return joined_values, joined_indices
        else:
            return joined_indices

    def join(self, other, on=None, how='left', lsuffix='', rsuffix='',
             sort=False, method='hash'):
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
        if how not in ['left', 'right', 'inner', 'outer']:
            raise NotImplementedError('unsupported {!r} join'.format(how))
        if on is not None:
            raise NotImplementedError('"on" is not supported yet')

        same_names = set(self.columns) & set(other.columns)
        if same_names and not (lsuffix or rsuffix):
            raise ValueError('there are overlapping columns but '
                             'lsuffix and rsuffix are not defined')

        return self._join(other=other, how=how, lsuffix=lsuffix,
                          rsuffix=rsuffix, sort=sort, same_names=same_names,
                          method=method)

    def _join(self, other, how, lsuffix, rsuffix, sort, same_names,
              method='hash', rightjoin=False):
        if how == 'right':
            # libgdf doesn't support right join directly, we will swap the
            # dfs and use left join
            return other._join(other=self, how='left', lsuffix=rsuffix,
                               rsuffix=lsuffix, sort=sort,
                               same_names=same_names, rightjoin=True)

        # Perform left, inner and outer join
        def fix_name(name, suffix):
            if name in same_names:
                return "{}{}".format(name, suffix)
            return name

        def gather_cols(outdf, indf, idx, joinidx, suffix):
            mask = (Series(idx) != -1).as_mask()
            for k in indf.columns:
                newcol = indf[k].take(idx).set_mask(mask).set_index(joinidx)
                outdf[fix_name(k, suffix)] = newcol

        def gather_empty(outdf, indf, idx, joinidx, suffix):
            for k in indf.columns:
                outdf[fix_name(k, suffix)] = indf[k][:0]

        lhs = self
        rhs = other

        df = DataFrame()

        joined_index, indexers = lhs.index.join(rhs.index, how=how,
                                                return_indexers=True,
                                                method=method)
        gather_fn = (gather_cols if len(joined_index) else gather_empty)
        lidx = indexers[0].to_gpu_array()
        ridx = indexers[1].to_gpu_array()

        # Gather columns
        left_args = (df, lhs, lidx, joined_index, lsuffix)
        right_args = (df, rhs, ridx, joined_index, rsuffix)
        args_order = ((right_args, left_args)
                      if rightjoin
                      else (left_args, right_args))
        for args in args_order:
            gather_fn(*args)

        # User requested a sort?
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
            Valid values are "sort", "hash", or "pygdf".
            "pygdf" method may be deprecated in the future, but is currently
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
        if (method == "pygdf"):
            from .groupby import Groupby
            if as_index:
                msg = "as_index==True not supported due to the lack of\
                    multi-index"
                raise NotImplementedError(msg)
            return Groupby(self, by=by)
        else:
            from .libgdf_groupby import LibGdfGroupby

            if as_index:
                msg = "as_index==True not supported due to the lack of\
                    multi-index"
                raise NotImplementedError(msg)
            return LibGdfGroupby(self, by=by, method=method)

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

        selected = Series(boolmask)
        newdf = DataFrame()
        for col in self.columns:
            newseries = self[col][selected]
            newdf[col] = newseries
        return newdf

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
