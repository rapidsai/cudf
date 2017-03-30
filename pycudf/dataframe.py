from __future__ import print_function, division

from collections import OrderedDict, Mapping

import numpy as np

from numba import cuda

from . import cudautils


class DataFrame(object):
    def __init__(self):
        self._cols = OrderedDict()
        self._size = 0

    def __getitem__(self, name):
        return SeriesView(self._cols[name])

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
        series = Series.from_any(data)
        self._sentry_column_size(len(series))
        self._cols[name] = series
        self._size = len(series)

    def add_row(self, **kwargs):
        if set(kwargs.keys()) != set(self.columns):
            raise ValueError("must provide all columns")

        for k, v in kwargs.items():
            self._cols[k].append(v)

        self._size += 1

    def flatten_columns(self):
        df = DataFrame()
        for k, cols in self._cols.items():
            df[k] = cols.flatten()
        return df

    def _sentry_flat(self, cols, opname='operation'):
        if any(col.buffer_count != 1 for col in cols):
            msg = '{} requires all columns to be flat'.format(opname)
            raise ValueError(msg)

    def as_matrix(self, columns, order='F'):
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
        self._sentry_flat(cols, 'as_matrix()')

        matrix = cuda.device_array(shape=(ncol, nrow), dtype=dtype, order='F')

        for colidx, inpcol in enumerate(cols):
            cudautils.copy_column(matrix, colidx, inpcol.as_gpu_array())

        return matrix.copy_to_host()


class Buffer(object):
    """A 1D gpu buffer.
    """
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

    def as_array(self):
        return self.as_gpu_array().copy_to_host()

    def as_gpu_array(self):
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
        sr = Series(size=buffer.size, dtype=buffer.dtype)
        sr._append(buffer)
        return sr

    @classmethod
    def from_array(cls, array):
        return cls.from_buffer(Buffer(array))

    def __init__(self, size, dtype):
        self._size = size
        self._dtype = dtype
        self._bufs = []

    def __len__(self):
        return self._size

    @property
    def buffer_count(self):
        return len(self._bufs)

    @property
    def dtype(self):
        return self._dtype

    def reserve(self, capacity):
        if not self._have_append_space_for(capacity):
            mem = cuda.device_array(shape=capacity, dtype=self.dtype)
            self._append(Buffer(mem, size=0, capacity=capacity))

    def append(self, arbitrary):
        series = Series.from_any(arbitrary)

        if len(series) < self.min_alloc_size:
            self.reserve(len(series))

        for buf in series._bufs:
            if self._have_append_space_for(buf.size):
                last_buffer = self._bufs[-1]
                last_buffer.extend(buf.as_gpu_array())
            else:
                self._append(buf.astype(self.dtype))
            self._size += buf.size

    def _have_append_space_for(self, needed):
        if self._bufs:
            last_buffer = self._bufs[-1]
            return last_buffer.avail_space >= needed
        else:
            return False

    def _append(self, buf):
            self._bufs.append(buf)

    def _sentry_single_buffer(self, opname='operation'):
        if self.buffer_count != 1:
            msg = '{} forbidden on multi-buffer Series'.format(opname)
            raise ValueError(msg)

    def as_array(self):
        """
        Sideeffect: combine all the buffers into one.
        """
        self._sentry_single_buffer('as_array')
        return self._bufs[0].as_array()

    def as_gpu_array(self):
        self._sentry_single_buffer('as_gpu_array')
        return self._bufs[0].as_gpu_array()

    def flatten(self):
        """
        Returns a single buffer series.
        """
        if self.buffer_count == 1:
            return self

        mem = cuda.device_array(shape=len(self), dtype=self.dtype)
        out = Buffer(mem, size=0, capacity=mem.size)
        for buf in self._bufs:
            out.extend(buf.as_gpu_array())
        return Series.from_buffer(out)


class SeriesView(object):
    _exported = ['flatten',
                 'as_array',
                 'as_gpu_array',
                 '__len__']

    def __init__(self, series):
        assert not isinstance(series, SeriesView)
        self._series = series

    def __getattr__(self, name):
        if name not in type(self)._exported:
            raise AttributeError(name)
        return getattr(self._series, name)


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

