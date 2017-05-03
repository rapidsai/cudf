from __future__ import print_function, division

from collections import OrderedDict

import numpy as np

from numba import cuda

from . import cudautils, utils


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
                         ('b', np.random.random(10))])
    >>> df2
      a b
    0 0 0.777831724018
    1 1 0.604480034669
    2 2 0.664111858618
    3 3 0.887777513028
    4 4 0.55838311246
    [5 more rows]
    """

    def __init__(self, name_series=None):
        self._size = 0
        self._cols = OrderedDict()
        # has initializer?
        if name_series is not None:
            for k, series in name_series:
                self.add_column(k, series)

    def __getitem__(self, name):
        """Access column by *name*
        """
        return self._cols[name]

    def __setitem__(self, name, col):
        """Add/set column by *name*
        """
        if name in self._cols:
            self._cols[name] = col
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
        if not self.columns:
            return ''
        if nrows is None:
            nrows = len(self)
        if ncols is None:
            ncols = len(self.columns)
        lastcol = None
        if len(self.columns) > ncols:
            cols = self.columns[:ncols - 1] + self.columns[-1:]
            lastcol = cols[-1]
        else:
            cols = self.columns
        # get values
        cells = OrderedDict()
        for c in cols:
            values = list(self[c][:nrows])
            cells[c] = ['' if v is None else str(v) for v in values]

        # compute column widths
        widths = {}
        for k, vs in cells.items():
            widths[k] = max(len(k), max(map(len, vs)))
        # format table
        out = []
        widthkey = len(str(nrows))
        header = [' ' * widthkey]
        header += ["{:{}}".format(k, widths[k]) for k in cols[:-1]]
        if lastcol is not None:
            header += ['...']
        header += ["{:{}}".format(k, widths[k]) for k in cols[-1:]]
        out.append(' '.join(header))
        for i in range(nrows):
            row = ["{:{}}".format(str(i), widthkey)]
            for k, vs in cells.items():
                if k == lastcol:
                    row.append('...')
                row.append("{:{}}".format(vs[i], widths[k]))
            out.append(' '.join(row))

        # show remiaining rows
        remaining_rows = len(self) - nrows
        if remaining_rows > 0:
            out.append("[{} more rows]".format(remaining_rows))

        # show remiaining cols
        if lastcol is not None:
            out.append("[{} more columns]".format(len(self.columns) - ncols))
        return '\n'.join(out)

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
                            ('b', list(range(20))),
                            ('c', list(range(20)))])
        >>> df[:4]   # get first 4 rows of all columns
          a b c
        0 0 0 0
        1 1 1 1
        2 2 2 2
        3 3 3 3
        4 4 4 4
        >>> df[-5:]  # get last 5 rows of all columns
          a  b  c
        0 15 15 15
        1 16 16 16
        2 17 17 17
        3 18 18 18
        4 19 19 19
        >>> df[:10, ['a', 'b']]   # get first 10 rows from 'a' and 'b' columns.
          a b
        0 0 0
        1 1 1
        2 2 2
        3 3 3
        4 4 4
        [5 more rows]
        """
        return Loc(self)

    @property
    def columns(self):
        """Returns a tuple of columns
        """
        return tuple(self._cols)

    def _sentry_column_size(self, size):
        if self._cols and self._size != size:
                raise ValueError('column size mismatch')

    def copy(self):
        "Shallow copy this dataframe"
        df = DataFrame()
        for k in self.columns:
            df[k] = self[k]
        return df

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
        series = Series.from_any(data)
        self._sentry_column_size(len(series))
        self._cols[name] = series
        self._size = len(series)

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
        dtype = cols[0]
        if any(dtype != c.dtype for c in cols):
            raise ValueError('all column must have the same dtype')
        for k, c in self._cols.items():
            if c.has_null_mask:
                raise ValueError("column {!r} is sparse".format(k))

        matrix = cuda.device_array(shape=(nrow, ncol), dtype=dtype, order="F")
        for colidx, inpcol in enumerate(cols):
            matrix[:, colidx].copy_to_device(inpcol.to_gpu_array())

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
            df[col] = self._df[col][row_slice]
        return df


class Buffer(object):
    """A 1D gpu buffer.
    """
    @classmethod
    def from_empty(cls, mem):
        return Buffer(mem, size=0, capacity=mem.size)

    def __init__(self, mem, size=None, capacity=None):
        if size is None:
            size = mem.size
        if capacity is None:
            capacity = size
        self.mem = cudautils.to_device(mem)
        _BufferSentry(self.mem).ndim(1)
        self.size = size
        self.capacity = capacity
        self.dtype = self.mem.dtype

    def __getitem__(self, arg):
        if isinstance(arg, slice):
            sliced = self.to_gpu_array()[arg]
            return Buffer(sliced)
        elif isinstance(arg, int):
            # normalize index
            if arg < 0:
                arg = self.size + arg
            if arg >= self.size:
                raise IndexError(arg)
            # getitem
            return self.mem[arg]
        else:
            raise NotImplementedError(type(arg))

    @property
    def avail_space(self):
        return self.capacity - self.size

    def _sentry_capacity(self, size_needed):
        if size_needed > self.avail_space:
            raise MemoryError('insufficient space in buffer')

    def append(self, element):
        self._sentry_capacity(1)
        self.extend(np.asarray(element, dtype=self.dtype))

    def extend(self, array):
        needed = array.size
        self._sentry_capacity(needed)
        array = cudautils.astype(array, dtype=self.dtype)
        self.mem[self.size:].copy_to_device(array)
        self.size += needed

    def astype(self, dtype):
        if self.dtype == dtype:
            return self
        else:
            return Buffer(cudautils.astype(self.mem, dtype=dtype))

    def to_array(self):
        return self.to_gpu_array().copy_to_host()

    def to_gpu_array(self):
        return self.mem[:self.size]


class Series(object):
    """
    Data and null-masks.

    ``Series`` objects are used as columns of ``DataFrame``.
    """
    @classmethod
    def from_any(cls, arbitrary):
        """Create Series from an arbitrary object

        Currently support inputs are:

        * ``Series``
        * ``Buffer``
        * numba device array
        * numpy array
        """
        if isinstance(arbitrary, Series):
            return arbitrary
        if isinstance(arbitrary, Buffer):
            return cls.from_buffer(arbitrary)
        elif cuda.devicearray.is_cuda_ndarray(arbitrary):
            return cls.from_array(arbitrary)
        else:
            if not isinstance(arbitrary, np.ndarray):
                arbitrary = np.asarray(arbitrary)
            return cls.from_array(arbitrary)

    @classmethod
    def from_buffer(cls, buffer):
        """Create a Series from a ``Buffer``
        """
        return Series(size=buffer.size, dtype=buffer.dtype, buffer=buffer)

    @classmethod
    def from_array(cls, array):
        """Create a Series from an array-like object.
        """
        return cls.from_buffer(Buffer(array))

    @classmethod
    def from_masked_array(cls, data, mask, null_count=None):
        """Create a Series with null-mask.

        Parameters
        ----------
        data : 1D array-like
            The values.  Null values must not be skipped.  They can appear
            as garbage values.
        mask : 1D array-like of numpy.uint8
            The null-mask.  Valid values are marked as ``1``; otherwise ``0``.
            The mask bit given the data index ``idx`` is computed as::

                (mask[idx // 8] >> (idx % 8)) & 1
        null_count : int, optional
            The number of null values.
            If None, it is calculated automatically.
        """

        dbuf = Buffer(data)
        mbuf = Buffer(mask)
        if null_count is None:
            _, nnz = cudautils.mask_assign_slot(dbuf.size, mbuf.mem)
            null_count = dbuf.size - nnz
        return cls(size=dbuf.size, dtype=dbuf.dtype, buffer=dbuf, mask=mbuf,
                   null_count=null_count)

    def __init__(self, size, dtype, buffer=None, mask=None, null_count=None):
        """
        Allocate a empty series with [size x dtype].
        The memory is uninitialized
        """
        self._size = size
        self._dtype = np.dtype(dtype)
        self._data = buffer
        self._mask = mask
        if null_count is None:
            if self._mask is not None:
                raise ValueError('null_count must be provided')
            null_count = 0
        self._null_count = null_count

    def __len__(self):
        """Returns the size of the ``Series`` including null values.
        """
        return self._size

    def __getitem__(self, arg):
        if isinstance(arg, slice):
            if self.null_count > 0:
                # compute mask slice
                start = arg.start if arg.start else 0
                stop = arg.stop if arg.stop else len(self)
                if arg.step is not None and arg.step != 1:
                    raise NotImplementedError(arg)
                maskslice = slice(utils.calc_chunk_size(start,
                                                        utils.mask_bitsize),
                                  utils.calc_chunk_size(stop,
                                                        utils.mask_bitsize))
                # slicing
                subdata = self._data.mem[arg]
                submask = self._mask.mem[maskslice]
                return self.from_masked_array(data=subdata, mask=submask)
            else:
                return self.from_buffer(self._data[arg])
        elif isinstance(arg, int):
            if self._mask is not None:
                valid = cudautils.mask_get.py_func(self._mask, arg)
            else:
                valid = 1
            return self._data[arg] if valid else None
        else:
            raise NotImplementedError(type(arg))

    @property
    def dtype(self):
        """dtype of the Series"""
        return self._dtype

    def append(self, arbitrary):
        """Append values from another ``Series`` or array-like object.
        Returns a new copy.
        """
        other = Series.from_any(arbitrary)
        newsize = len(self) + len(other)
        # allocate memory
        mem = cuda.device_array(shape=newsize, dtype=self._dtype)
        newbuf = Buffer.from_empty(mem)
        # copy into new memory
        for buf in [self._data, other._data]:
            newbuf.extend(buf.to_gpu_array())
        # return new series
        return self.from_any(newbuf)

    @property
    def null_count(self):
        """Number of null values"""
        return self._null_count

    @property
    def has_null_mask(self):
        """A boolean indicating whether a null-mask is needed"""
        return self.null_count > 0

    def fillna(self, value):
        """Fill null values with ``value``.

        Returns a copy with null filled.
        """
        if not self.has_null_mask:
            return self
        out = cudautils.fillna(data=self._data.to_gpu_array(),
                               mask=self._mask.to_gpu_array(),
                               value=value)
        return self.from_array(out)

    def to_dense_buffer(self):
        """Get dense (no null values) ``Buffer`` of the data.

        Notes
        -----

        Null values are skipped.  Therefore, the output size could be smaller.
        """
        if self.has_null_mask:
            return self._copy_to_dense_buffer()
        else:
            return self._data

    def _copy_to_dense_buffer(self):
        data = self._data.to_gpu_array()
        mask = self._mask.to_gpu_array()
        nnz, mem = cudautils.copy_to_dense(data=data, mask=mask)
        return Buffer(mem, size=nnz, capacity=mem.size)

    def to_array(self):
        """Get a dense numpy array for the data.
        """
        return self.to_dense_buffer().to_array()

    def to_gpu_array(self):
        """Get a dense numba device array for the data.
        """
        return self.to_dense_buffer().to_gpu_array()

    @property
    def data(self):
        """The gpu buffer for the data
        """
        return self._data

    @property
    def nullmask(self):
        """The gpu buffer for the null-mask
        """
        if self.has_null_mask:
            return self._mask
        else:
            raise ValueError('Series has no null mask')

    def astype(self, dtype):
        """Convert to the given ``dtype``.

        Returns
        -------
        If the dtype changed, a new ``Series`` is returned by casting each
        values to the given dtype.
        If the dtype is not changed, ``self`` is returned.
        """
        if dtype == self.dtype:
            return self
        return Series.from_buffer(self.data.astype(dtype))

    def one_hot_encoding(self, cats, dtype='float64'):
        """Perform one-hot-encoding

        Parameters
        ----------
        cats : sequence of values
                values representing each category.
        dtype : numpy.dtype
                specifies the output dtype.

        Returns
        -------
        A sequence of new series for each category.  Its length is determined
        by the length of ``cats``.
        """
        if self.dtype.kind not in 'iuf':
            raise TypeError('expecting integer or float dtype')

        dtype = np.dtype(dtype)
        out = []
        for cat in cats:
            buf = cudautils.apply_equal_constant(arr=self.to_gpu_array(),
                                                 val=cat, dtype=dtype)
            out.append(Series.from_array(buf))
        return out

    #
    # Stats
    #

    def min(self):
        """Compute the min of the series
        """
        arr = self.to_dense_buffer().to_gpu_array()
        maxval = utils.get_numeric_type_info(self.dtype).max
        return cudautils.compute_min(arr, init=maxval)

    def max(self):
        """Compute the max of the series
        """
        arr = self.to_dense_buffer().to_gpu_array()
        minval = utils.get_numeric_type_info(self.dtype).min
        return cudautils.compute_max(arr, init=minval)

    def mean(self):
        """Compute the mean of the series
        """
        arr = self.to_dense_buffer().to_gpu_array()
        return cudautils.compute_mean(arr)

    def std(self):
        """Compute the standard deviation of the series
        """
        return np.sqrt(self.var())

    def var(self):
        """Compute the variance of the series
        """
        arr = self.to_dense_buffer().to_gpu_array()
        mu, var = cudautils.compute_stats(arr)
        return var

    def mean_var(self):
        """Compute mean and variance at the same time.
        """
        arr = self.to_dense_buffer().to_gpu_array()
        mu, var = cudautils.compute_stats(arr)
        return mu, var

    def unique_k(self, k):
        """Returns a list of at most k unique values.
        """
        if self.null_count == len(self):
            return np.empty(0, dtype=self.dtype)
        arr = self.to_dense_buffer().to_gpu_array()
        return cudautils.compute_unique_k(arr, k=k)

    def scale(self):
        """Scale values to [0, 1] in float64
        """
        if self.null_count != 0:
            msg = 'masked series not supported by this operation'
            raise NotImplementedError(msg)
        vmin = self.min()
        vmax = self.max()
        gpuarr = self.to_gpu_array()
        scaled = cudautils.compute_scale(gpuarr, vmin, vmax)
        return Series.from_array(scaled)


class BufferSentryError(ValueError):
    pass


class _BufferSentry(object):
    def __init__(self, buf):
        self._buf = buf

    def dtype(self, dtype):
        if self._buf.dtype != dtype:
            raise BufferSentryError('dtype mismatch')
        return self

    def ndim(self, ndim):
        if self._buf.ndim != ndim:
            raise BufferSentryError('ndim mismatch')
        return self

    def contig(self):
        if not self._buf.is_c_contiguous():
            raise BufferSentryError('non contiguous')


def _make_mask(size):
    size = utils.calc_chunk_size(size, utils.mask_bitsize)
    return cuda.device_array(shape=size, dtype=utils.mask_dtype)


def _make_mask_from_stride(size, stride):
    mask = _make_mask(size)
    cudautils.set_mask_from_stride(mask=mask, stride=stride)
    return mask

