from __future__ import print_function, division

from collections import OrderedDict, Mapping

import numpy as np

from numba import cuda

from . import cudautils, utils


class DataFrame(object):
    def __init__(self):
        self._cols = OrderedDict()
        self._size = 0

    def __getitem__(self, name):
        return self._cols[name]

    def __setitem__(self, name, col):
        self.add_column(name, col)

    def __len__(self):
        return self._size

    @property
    def columns(self):
        return tuple(self._cols)

    def _sentry_column_size(self, size):
        if self._cols and self._size != size:
                raise ValueError('column size mismatch')

    def add_column(self, name, data):
        if name in self._cols:
            raise NameError('duplicated column name {!r}'.format(name))
        series = Series.from_any(data)
        self._sentry_column_size(len(series))
        self._cols[name] = series
        self._size = len(series)

    def concat(self, *dfs):
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

    def as_matrix(self, columns=None):
        """
        Returns a (nrow x ncol) numpy ndarray in "F" order.
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

        return matrix.copy_to_host()


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
        _BufferSentry(self.mem).ndim(1).contig()
        self.size = size
        self.capacity = capacity
        self.dtype = self.mem.dtype

    def __getitem__(self, arg):
        if isinstance(arg, slice):
            start = arg.start or 0
            stop = arg.stop or -1
            step = arg.step
            assert step is None

            if stop < 0:
                stop = self.size

            sliced = self.mem[start:stop]
            size = stop - start
            return Buffer(mem=sliced, size=size)

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
    Data and null-masks are stored as List[Array].

    """
    min_alloc_size = 32

    @classmethod
    def from_any(cls, arbitrary):
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
        return Series(size=buffer.size, dtype=buffer.dtype, buffer=buffer)

    @classmethod
    def from_array(cls, array):
        return cls.from_buffer(Buffer(array))

    def __init__(self, size, dtype, buffer=None, mask=None):
        """
        Allocate a empty series with [size x dtype].
        The memory is uninitialized
        """
        self._size = size
        self._dtype = np.dtype(dtype)
        self._data = buffer
        self._mask = mask

    def __len__(self):
        return self._size

    def __getitem__(self, arg):
        if isinstance(arg, slice):
            if arg.step is not None:
                dataslice = slice(arg.start, arg.stop, None)
                data = self._data[dataslice]
                maskmem = _make_mask_from_stride(size=data.size,
                                                 stride=arg.step)
                mask = Buffer(maskmem)
                sr = Series(size=data.size, dtype=data.dtype,  buffer=data,
                            mask=mask)
                return sr
            else:
                return self.from_buffer(self._data[arg])
        else:
            raise NotImplementedError(type(arg))

    @property
    def dtype(self):
        return self._dtype

    def append(self, arbitrary):
        """
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
    def has_null_mask(self):
        return self._mask is not None

    def to_dense_buffer(self):
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
        return self.to_dense_buffer().to_array()

    def to_gpu_array(self):
        return self.to_dense_buffer().to_gpu_array()

    @property
    def data(self):
        """
        The gpu buffer for the data
        """
        return self._data

    @property
    def nullmask(self):
        """
        The gpu buffer for the null-mask
        """
        if self.has_null_mask:
            return self._mask
        else:
            raise ValueError('Series has no null mask')


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

