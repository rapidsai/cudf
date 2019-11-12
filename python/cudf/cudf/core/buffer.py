import functools
import operator

import numpy as np

import nvstrings
from rmm import DeviceBuffer


class Buffer:
    def __init__(self, ptr, size=None, owner=None):
        if not ptr:
            self.ptr = 0
            self.size = 0
        else:
            self.ptr = ptr
            self.size = size
        self._owner = owner

    @classmethod
    def from_array_like(cls, data):
        if hasattr(data, "__cuda_array_interface__"):
            ptr, size = _buffer_data_from_array_interface(
                data.__cuda_array_interface__
            )
            return Buffer(ptr, size, owner=data)
        elif hasattr(data, "__array_interface__"):
            ptr, size = _buffer_data_from_array_interface(
                data.__array_interface__
            )
            dbuf = DeviceBuffer(ptr, size)
            return Buffer.from_device_buffer(dbuf)
        else:
            raise TypeError(
                f"Cannot construct Buffer from {data.__class__.__name__}"
            )

    @classmethod
    def from_device_buffer(cls, dbuf):
        assert isinstance(dbuf, DeviceBuffer)
        return Buffer(dbuf.ptr, dbuf.size, owner=dbuf)

    @classmethod
    def empty(cls, size):
        dbuf = DeviceBuffer(size=size)
        return Buffer.from_device_buffer(dbuf)


def _buffer_data_from_array_interface(array_interface):
    ptr = array_interface["data"][0]
    itemsize = np.dtype(array_interface["typestr"]).itemsize
    size = functools.reduce(operator.mul, array_interface["shape"])
    return ptr, size * itemsize
