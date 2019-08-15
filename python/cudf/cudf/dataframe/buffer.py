import pickle

import numba.cuda
import numpy as np

from librmm_cffi import librmm as rmm

from cudf.utils import cudautils, utils


class Buffer(object):
    """A 1D gpu buffer.
    """

    _cached_ipch = None

    @classmethod
    def from_empty(cls, mem, size=0):
        """From empty device array
        """
        return cls(mem, size=size, capacity=mem.size)

    @classmethod
    def null(cls, dtype):
        """Create a "null" buffer with a zero-sized device array.
        """
        mem = rmm.device_array(0, dtype=dtype)
        return cls(mem, size=0, capacity=0)

    def __init__(
        self, mem, size=None, capacity=None, categorical=False, header=None
    ):
        if size is None:
            if categorical:
                size = len(mem)
            elif hasattr(mem, "__len__"):
                if hasattr(mem, "ndim") and mem.ndim == 0:
                    pass
                elif len(mem) == 0:
                    size = 0
            if hasattr(mem, "size"):
                size = mem.size
        if capacity is None:
            capacity = size
        # memoryviews can come from UCX when the length of the DataFrame
        # is 0 -- for example: joins resulting in empty frames or metadata
        if isinstance(mem, memoryview):
            mem = np.frombuffer(mem, dtype=header["dtype"])
            size = mem.size
        if not (
            isinstance(mem, np.ndarray)
            or numba.cuda.driver.is_device_memory(mem)
        ):
            # this is probably a ucp_py.BufferRegion memory object
            # check the header for info -- this should be encoded from
            # serialization process.  Lastly, `typestr` and `shape` *must*
            # manually set *before* consuming the buffer as a DeviceNDArray
            mem.typestr = header.get("dtype", "B")
            mem.shape = header.get("shape", len(mem))
            size = mem.shape[0]
        self.mem = cudautils.to_device(mem)
        _BufferSentry(self.mem).ndim(1)
        self.size = size
        self.capacity = capacity
        self.dtype = self.mem.dtype

    def serialize(self):
        """Called when dask.distributed is performing a serialization on this
        object.

        Do not use this directly.  It is invoked by dask.distributed.

        Parameters
        ----------
        serialize : callable
             Used to serialize data that needs serialization .
        context : dict; optional
            If not ``None``, it contains information about the destination.

        Returns
        -------
        (header, frames)
            See custom serialization documentation in dask.distributed.
        """
        header = {}
        header["type"] = pickle.dumps(type(self))
        header["shape"] = self.mem.shape
        header["strides"] = self.mem.strides
        header["dtype"] = self.mem.dtype.str

        return header, [self.mem]

    @classmethod
    def deserialize(cls, header, frames):
        """Called when dask.distributed is performing a deserialization for
        data of this class.

        Do not use this directly.  It is invoked by dask.distributed.

        Parameters
        ----------

        deserialize : callable
             Used to deserialize data that needs further deserialization .
        header, frames : dict
            See custom serialization documentation in dask.distributed.

        Returns
        -------
        obj : Buffer
            Returns an instance of Buffer.
        """
        return Buffer(frames[0], header=header)

    def __reduce__(self):
        cpumem = self.to_array()
        # Note: pickled Buffer only stores *size* element.
        return type(self), (cpumem,)

    def __sizeof__(self):
        return int(self.mem.alloc_size)

    def __getitem__(self, arg):
        if isinstance(arg, slice):
            sliced = self.mem[arg]
            buf = Buffer(sliced)
            buf.dtype = self.dtype  # for np.datetime64 support
            return buf
        elif isinstance(arg, (int, np.integer)):
            arg = utils.normalize_index(int(arg), self.size)
            item = self.mem[arg]
            if isinstance(item, str):
                return item
            # the dtype argument is necessary for datetime64 support
            # because currently we can't pass datetime64 types into
            # cuda dev arrays, so the type of the cuda dev array is
            # an i64, and we view it as the dtype on the buffer
            return item.view(self.dtype)
        else:
            raise NotImplementedError(type(arg))

    @property
    def avail_space(self):
        return self.capacity - self.size

    def _sentry_capacity(self, size_needed):
        if size_needed > self.avail_space:
            raise MemoryError("insufficient space in buffer")

    def append(self, element):
        self._sentry_capacity(1)
        self.extend(np.asarray(element, dtype=self.dtype))

    def extend(self, array):
        from cudf.dataframe import columnops

        needed = array.size
        self._sentry_capacity(needed)

        array = columnops.as_column(array).astype(self.dtype).data.mem

        self.mem[self.size : self.size + needed].copy_to_device(array)
        self.size += needed

    def to_array(self):
        return self.to_gpu_array().copy_to_host()

    def to_gpu_array(self):
        return self.mem[: self.size]

    def copy(self):
        """Deep copy the buffer
        """
        return Buffer(
            mem=cudautils.copy_array(self.mem),
            size=self.size,
            capacity=self.capacity,
        )

    def as_contiguous(self):
        out = Buffer(
            mem=cudautils.as_contiguous(self.mem),
            size=self.size,
            capacity=self.capacity,
        )
        assert out.is_contiguous()
        return out

    def is_contiguous(self):
        return self.mem.is_c_contiguous()

    def astype(self, dtype):
        from cudf.dataframe import columnops

        return columnops.as_column(self).astype(dtype).data


class BufferSentryError(ValueError):
    pass


class _BufferSentry(object):
    def __init__(self, buf):
        self._buf = buf

    def dtype(self, dtype):
        if self._buf.dtype != dtype:
            raise BufferSentryError("dtype mismatch")
        return self

    def ndim(self, ndim):
        if self._buf.ndim != ndim:
            raise BufferSentryError("ndim mismatch")
        return self

    def contig(self):
        if not self._buf.is_c_contiguous():
            raise BufferSentryError("non contiguous")
