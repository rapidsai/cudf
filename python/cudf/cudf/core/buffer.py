import functools
import operator
import pickle

import numpy as np

import rmm
from rmm import DeviceBuffer

from cudf.core.abc import Serializable


class Buffer(Serializable):
    def __init__(self, data=None, size=None, owner=None):
        """
        A Buffer represents a device memory allocation.

        Parameters
        ----------
        data : Buffer, array_like, int
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
            self._owner = owner or data._owner
        elif hasattr(data, "__array_interface__") or hasattr(
            data, "__cuda_array_interface__"
        ):

            self._init_from_array_like(data, owner)
        elif isinstance(data, memoryview):
            self._init_from_array_like(np.asarray(data), owner)
        elif isinstance(data, int):
            if not isinstance(size, int):
                raise TypeError("size must be integer")
            self.ptr = data
            self.size = size
            self._owner = owner
        elif data is None:
            self.ptr = 0
            self.size = 0
            self._owner = None
        else:
            try:
                data = memoryview(data)
            except TypeError:
                raise TypeError("data must be Buffer, array-like or integer")
            self._init_from_array_like(np.asarray(data), owner)

    def __len__(self):
        return self.size

    @property
    def nbytes(self):
        return self.size

    @property
    def __cuda_array_interface__(self):
        intf = {
            "data": (self.ptr, False),
            "shape": (self.size,),
            "strides": (1,),
            "typestr": "|u1",
            "version": 0,
        }
        return intf

    def to_host_array(self):
        data = np.empty((self.size,), "u1")
        rmm._lib.device_buffer.copy_ptr_to_host(self.ptr, data)
        return data

    def _init_from_array_like(self, data, owner):
        if hasattr(data, "__cuda_array_interface__"):
            ptr, size = _buffer_data_from_array_interface(
                data.__cuda_array_interface__
            )
            self.ptr = ptr
            self.size = size
            self._owner = owner or data
        elif hasattr(data, "__array_interface__"):
            ptr, size = _buffer_data_from_array_interface(
                data.__array_interface__
            )
            dbuf = DeviceBuffer(ptr=ptr, size=size)
            self._init_from_array_like(dbuf, owner)
        else:
            raise TypeError(
                f"Cannot construct Buffer from {data.__class__.__name__}"
            )

    def serialize(self):
        header = {}
        header["type-serialized"] = pickle.dumps(type(self))
        header["constructor-kwargs"] = {}
        header["desc"] = self.__cuda_array_interface__.copy()
        frames = [self]
        return header, frames

    @classmethod
    def deserialize(cls, header, frames):
        buf = cls(frames[0], **header["constructor-kwargs"])

        if header["desc"]["shape"] != buf.__cuda_array_interface__["shape"]:
            raise ValueError(
                "Recieved a `Buffer` with the wrong size."
                " Expected {0}, but got {1}".format(
                    header["desc"]["shape"],
                    buf.__cuda_array_interface__["shape"],
                )
            )

        return buf

    @classmethod
    def empty(cls, size):
        dbuf = DeviceBuffer(size=size)
        return Buffer(dbuf)


def _buffer_data_from_array_interface(array_interface):
    ptr = array_interface["data"][0]
    if ptr is None:
        ptr = 0
    itemsize = np.dtype(array_interface["typestr"]).itemsize
    shape = (
        array_interface["shape"] if len(array_interface["shape"]) > 0 else (1,)
    )
    size = functools.reduce(operator.mul, shape)
    return ptr, size * itemsize
