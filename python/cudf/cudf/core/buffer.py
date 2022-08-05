# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from __future__ import annotations

import functools
import operator
import pickle
from typing import (
    Any,
    Dict,
    Mapping,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np

import rmm

import cudf
from cudf.core.abc import Serializable


@runtime_checkable
class DeviceBufferLike(Protocol):
    def __len__(self) -> int:
        ...

    @property
    def ptr(self) -> int:
        ...

    @property
    def size(self) -> int:
        ...

    @property
    def owner(self) -> object:
        ...

    @property
    def __cuda_array_interface__(self) -> Mapping:
        ...

    def serialize(self) -> Tuple[dict, list]:
        ...

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> DeviceBufferLike:
        ...


def as_device_buffer_like(
    obj: Any, size: int = None, offset: int = 0
) -> DeviceBufferLike:
    """
    Factory function to wrap `obj` in a DeviceBufferLike object.

    If `obj` isn't device-buffer-like already, a new buffer that implements
    DeviceBufferLike and points to the memory of `obj` is created. If `obj`
    represents host memory, it is copied to a new `rmm.DeviceBuffer` device
    allocation. Otherwise, the data of `obj` is **not** copied.

    The returned Buffer keeps a reference to `obj` in order to retain the
    lifetime of `obj`.

    If `size` and/or `offset` is specified, `obj` must be device-buffer-like.

    Raises ValueError if the data of `obj` isn't C-contiguous.

    Parameters
    ----------
    obj : buffer-like or array-like
        An object that exposes either device or host memory through
        `__array_interface__`, `__cuda_array_interface__`, or the
        buffer protocol. Only when `obj` represents host memory are
        data copied.
    size : int, optional
        Size of buffer in bytes.
    offset : int, optional
        Start offset relative to `obj.ptr` (in bytes).

    Return
    ------
    DeviceBufferLike
        A device-buffer-like instance that represents the device memory
        of `obj`.
    """
    if isinstance(obj, DeviceBufferLike):
        size = obj.size - offset if size is None else size
        if size == obj.size and offset == 0:
            return obj
        return Buffer(
            data=obj.ptr + offset,
            size=size,
            owner=obj,
        )
    elif size or offset:
        raise ValueError(
            "`obj` must be DeviceBufferLike when `size` and/or `offset`"
            "is specified"
        )
    return Buffer(obj)


class Buffer(Serializable):
    """
    A Buffer represents device memory.

    Usually Buffers will be created using `as_device_buffer_like(obj)`,
    which will make sure that `obj` is device-buffer-like and not a `Buffer`
    necessarily.

    Parameters
    ----------
    data : int or buffer-like or array-like
        An integer representing a pointer to device memory or a buffer-like
        ot array-like object. When not an integer, `size` and `owner` must
        be None.
    size : int, optional
        Size of device memory in bytes. Must be specified when `ptr` is a
        integer.
    owner : object, optional
        Python object to which the lifetime of the memory allocation is tied.
        A reference to this object is kept in the returned Buffer.
    """

    _ptr: int
    _size: int
    _owner: object

    def __init__(
        self, data: Union[int, Any], size: int = None, owner: object = None
    ):
        if isinstance(data, int):
            if size is None:
                raise ValueError(
                    "size must be specified when `data` is an integer"
                )
            self._ptr = data
            self._size = size
            self._owner = owner
        else:
            if size is not None or owner is not None:
                raise ValueError(
                    "`size` and `owner` must be None when "
                    "`data` is an buffer-like object"
                )

            # `data` is a buffer-like object
            buf: Any = data
            if isinstance(buf, rmm.DeviceBuffer):
                self._ptr, self._size, self._owner = buf.ptr, buf.size, buf
                return
            iface = getattr(buf, "__cuda_array_interface__", None)
            if iface:
                ptr, size = _get_ptr_and_size(iface)
                self._ptr, self._size, self._owner = ptr, size, buf
                return
            ptr, size = _get_ptr_and_size(np.asarray(buf).__array_interface__)
            buf = rmm.DeviceBuffer(ptr=ptr, size=size)
            self._ptr, self._size, self._owner = buf.ptr, buf.size, buf

    def __len__(self) -> int:
        return self._size

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def size(self) -> int:
        return self._size

    @property
    def owner(self) -> object:
        return self._owner

    @property
    def nbytes(self) -> int:
        return self._size

    @property
    def __cuda_array_interface__(self) -> dict:
        return {
            "data": (self.ptr, False),
            "shape": (self.size,),
            "strides": None,
            "typestr": "|u1",
            "version": 0,
        }

    def to_host_array(self):
        data = np.empty((self.size,), "u1")
        rmm._lib.device_buffer.copy_ptr_to_host(self.ptr, data)
        return data

    def serialize(self) -> Tuple[dict, list]:
        header = {}  # type: Dict[Any, Any]
        header["type-serialized"] = pickle.dumps(type(self))
        header["constructor-kwargs"] = {}
        header["desc"] = self.__cuda_array_interface__.copy()
        header["desc"]["strides"] = (1,)
        header["frame_count"] = 1
        frames = [self]
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> Buffer:
        assert (
            header["frame_count"] == 1
        ), "Only expecting to deserialize Buffer with a single frame."
        buf = cls(frames[0], **header["constructor-kwargs"])

        if header["desc"]["shape"] != buf.__cuda_array_interface__["shape"]:
            raise ValueError(
                f"Received a `Buffer` with the wrong size."
                f" Expected {header['desc']['shape']}, "
                f"but got {buf.__cuda_array_interface__['shape']}"
            )

        return buf


def _get_ptr_and_size(array_interface: Mapping) -> Tuple[int, int]:
    """
    Return the pointer and size of an array interface.

    Raises ValueError if array isn't C-contiguous
    """

    def is_c_contiguous(shape, strides, itemsize):
        if strides is None or any(dim == 0 for dim in shape):
            return True
        sd = itemsize
        for dim, stride in zip(reversed(shape), reversed(strides)):
            if dim > 1 and stride != sd:
                return False
            sd *= dim
        return True

    shape = array_interface["shape"] or (1,)
    itemsize = cudf.dtype(array_interface["typestr"]).itemsize
    ptr = array_interface["data"][0] or 0
    if not is_c_contiguous(shape, array_interface["strides"], itemsize):
        raise ValueError("Buffer data must be 1D C-contiguous")
    size = functools.reduce(operator.mul, shape)
    return ptr, size * itemsize
