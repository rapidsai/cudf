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
    @classmethod
    def from_buffer(
        cls, buffer: DeviceBufferLike, size: int = None, offset: int = 0
    ):
        ...

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


def as_device_buffer_like(obj: object) -> DeviceBufferLike:
    """
    Factory function to wrap `obj` in a DeviceBufferLike object.

    If `obj` isn't device-buffer-like already, a new buffer that implements
    DeviceBufferLike and points to the memory of `obj` is created. If `obj`
    represents host memory, it is copied to a new `rmm.DeviceBuffer` device
    allocation. Otherwise, the data of `obj` is **not** copied.

    The returned Buffer keeps a reference to `obj` in order to retain the
    lifetime of `obj`.

    Raises ValueError if the data of `obj` isn't C-contiguous.

    Parameters
    ----------
    obj : buffer-like
        An object that exposes either device or host memory through
        `__array_interface__`, `__cuda_array_interface__`, or the
        buffer protocol. Only when `obj` represents host memory are
        data copied.

    Return
    ------
    Buffer
        A Buffer instance that represents the device memory of `obj`
    """
    if isinstance(obj, Buffer):
        return obj
    return Buffer(obj)


def buffer_from_pointer(ptr: int, size: int, owner: object) -> Buffer:
    """
    Factory function to wrap a device memory pointer in a Buffer object.

    Never copies any data and `ptr` must represents device memory.

    The returned Buffer keeps a reference to `obj` in order to
    retain the lifetime of `obj`.

    Parameters
    ----------
    ptr : int
        An integer representing a pointer to device memory.
    size : int
        Size of device memory in bytes.
    owner : object
        Python object to which the lifetime of the memory allocation is tied.
        A reference to this object is kept in the returned Buffer.

    Return
    ------
    Buffer
        A Buffer instance that represents the device memory of `ptr`
    """
    return Buffer(ptr, size, owner)


class Buffer(Serializable):
    """
    A Buffer represents a device memory allocation.

    Usually Buffers will be created using `as_device_buffer_like(obj)`,
    which will make sure that `obj` is device-buffer-like and not a `Buffer`
    necessarily.

    Parameters
    ----------
    ptr : int or buffer-like
        An integer representing a pointer to device memory or a buffer-like
        object. When a buffer-like object is given, `size` and `owner` must
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
        self, ptr: Union[int, Any], size: int = None, owner: object = None
    ):
        if isinstance(ptr, int):
            if size is None:
                raise ValueError(
                    "size must be specified when `ptr` is an integer"
                )
            self._ptr = ptr
            self._size = size
            self._owner = owner
        else:
            if size is not None or owner is not None:
                raise ValueError(
                    "`size` and `owner` must be None when "
                    "`ptr` is an buffer-like object"
                )

            # `ptr` is a buffer-like object
            obj: Any = ptr
            if isinstance(obj, rmm.DeviceBuffer):
                self._ptr, self._size, self._owner = obj.ptr, obj.size, obj
                return
            iface = getattr(obj, "__cuda_array_interface__", None)
            if iface:
                ptr, size = _get_ptr_and_size(iface)
                self._ptr, self._size, self._owner = ptr, size, obj
                return
            ptr, size = _get_ptr_and_size(np.asarray(obj).__array_interface__)
            obj = rmm.DeviceBuffer(ptr=ptr, size=size)
            self._ptr, self._size, self._owner = obj.ptr, obj.size, obj

    @classmethod
    def from_buffer(
        cls, buffer: DeviceBufferLike, size: int = None, offset: int = 0
    ):
        """
        Create a buffer from another buffer

        Parameters
        ----------
        buffer : Buffer
            The base buffer, which will also be set as the owner of
            the memory allocation.
        size : int, optional
            Size of the memory allocation (default: `buffer.size`).
        offset : int, optional
            Start offset relative to `buffer.ptr`.
        """
        return cls(
            ptr=buffer.ptr + offset,
            size=buffer.size if size is None else size,
            owner=buffer,
        )

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
