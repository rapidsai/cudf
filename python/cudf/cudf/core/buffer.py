# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from __future__ import annotations

import functools
import operator
import pickle
from typing import Any, Dict, Mapping, Tuple, Union

import numpy as np

import rmm

import cudf
from cudf.core.abc import Serializable


def as_buffer(obj: object) -> Buffer:
    """
    Factory function to wrap `obj` in a Buffer object.

    If `obj` isn't a Buffer already, a new Buffer that points to the
    memory of `obj` is created. If `obj` represents host memory, it is
    copied to a new `rmm.DeviceBuffer` device allocation. Otherwise,
    the data of `obj` is **not** copied.

    The returned Buffer keeps a reference to `obj` in order to
    retain the lifetime of `obj`.

    Raises ValueError if the data of `obj` isn't c-contiguous.

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
    ret = Buffer.__new__(Buffer)
    _init_buffer_from_any(ret, obj)
    return ret


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

    Usually a Buffer instance should be created through factory functions
    such as `as_buffer()` and `buffer_from_pointer()`. However, for backward
    compatibility, it is possible to instantiate a Buffer directly given a
    buffer-like object, which is equivalent to calling `as_buffer()`.

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
        self, ptr: Union[int, object], size: int = None, owner: object = None
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
            _init_buffer_from_any(self, ptr)

    @classmethod
    def from_buffer(cls, buffer: Buffer, size: int = None, offset: int = 0):
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
            ptr=buffer._ptr + offset,
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

    Raises ValueError if array isn't c-contiguous
    """

    def is_c_contiguous(shape, strides, itemsize):
        ndim = len(shape)
        assert strides is None or ndim == len(strides)

        if (
            ndim == 0
            or strides is None
            or (ndim == 1 and strides[0] == itemsize)
        ):
            return True

        # any dimension zero, trivial case
        for dim in shape:
            if dim == 0:
                return True

        for this_dim, this_stride in zip(shape, strides):
            if this_stride != this_dim * itemsize:
                return False
        return True

    shape = array_interface["shape"] or (1,)
    itemsize = cudf.dtype(array_interface["typestr"]).itemsize
    ptr = array_interface["data"][0] or 0
    if not is_c_contiguous(shape, array_interface["strides"], itemsize):
        raise ValueError("Buffer data must be 1D C-contiguous")
    size = functools.reduce(operator.mul, shape)
    return ptr, size * itemsize


def _init_buffer_from_any(buf: Buffer, obj: object) -> None:
    """
    Initiate `buf` based on `obj`.

    Parameters
    ----------
    buf : Buffer
        The buffer to initiate.
    obj : buffer-like
        An object that exposes either device or host memory through
        `__array_interface__`, `__cuda_array_interface__`, or the
        buffer protocol. Only when `obj` represents host memory are
        data copied.
    """

    if isinstance(obj, rmm.DeviceBuffer):
        buf._ptr, buf._size, buf._owner = obj.ptr, obj.size, obj
        return
    iface = getattr(obj, "__cuda_array_interface__", None)
    if iface:
        ptr, size = _get_ptr_and_size(iface)
        buf._ptr, buf._size, buf._owner = ptr, size, obj
        return
    host_ary = np.asarray(obj)
    ptr, size = _get_ptr_and_size(host_ary.__array_interface__)
    _init_buffer_from_any(buf, rmm.DeviceBuffer(ptr=ptr, size=size))
