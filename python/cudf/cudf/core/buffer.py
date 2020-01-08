import functools
import operator

import numpy as np

import rmm
from rmm import DeviceBuffer, _DevicePointer


class Buffer:
    def __init__(self, data=None, size=None, owner=None):
        """
        A Buffer represents a device memory allocation.

        Parameters
        ----------
        data : array_like, int
            An array-like object or integer representing a
            device or host pointer to pre-allocated memory.
        size : int, optional
            Size of memory allocation. Required if a pointer
            is passed for `data`.
        owner : object, optional
            Python object to which the lifetime of the memory
            allocation is tied. If provided, a reference to this
            object is kept in this Buffer.
        """
        if isinstance(data, Buffer):
            self.ptr = data.ptr
            self.size = data.size
            self._owner = owner or data
        elif isinstance(data, _DevicePointer):
            self.ptr = data.ptr
            self.size = size
            self._owner = data
        elif hasattr(data, "__array_interface__") or hasattr(
            data, "__cuda_array_interface__"
        ):

            self._init_from_array_like(data)
        elif isinstance(data, memoryview):
            self._init_from_array_like(np.asarray(data))
        else:
            if data is None:
                self.ptr = 0
                self.size = 0
            else:
                self.ptr = data
                self.size = size
            self._owner = owner

    def __reduce__(self):
        return self.__class__, (self.to_host_array(),)

    def to_host_array(self):
        return rmm.device_array_from_ptr(
            self.ptr, nelem=self.size, dtype="int8"
        ).copy_to_host()

    def _init_from_array_like(self, data):
        if hasattr(data, "__cuda_array_interface__"):
            ptr, size = _buffer_data_from_array_interface(
                data.__cuda_array_interface__
            )
            self.ptr = ptr
            self.size = size
            self._owner = data
        elif hasattr(data, "__array_interface__"):
            ptr, size = _buffer_data_from_array_interface(
                data.__array_interface__
            )
            dbuf = DeviceBuffer(ptr=ptr, size=size)
            self._init_from_array_like(dbuf)
        else:
            raise TypeError(
                f"Cannot construct Buffer from {data.__class__.__name__}"
            )

    @classmethod
    def empty(cls, size):
        dbuf = DeviceBuffer(ptr=None, size=size)
        return Buffer(dbuf)


def _buffer_data_from_array_interface(array_interface):
    ptr = array_interface["data"][0]
    itemsize = np.dtype(array_interface["typestr"]).itemsize
    size = functools.reduce(operator.mul, array_interface["shape"])
    return ptr, size * itemsize
