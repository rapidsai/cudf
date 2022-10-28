# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from __future__ import annotations

import math
import pickle
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import numpy

import rmm

import cudf
from cudf.core.abc import Serializable
from cudf.utils.string import format_bytes


class Buffer(Serializable):
    """A Buffer represents device memory.

    Usually buffers will be created using `as_buffer(obj)`,
    which will make sure that `obj` is buffer and not a `Buffer`
    necessarily.

    Parameters
    ----------
    data : int or buffer-like or array-like
        An integer representing a pointer to device memory or a buffer-like
        or array-like object. When not an integer, `size` and `owner` must
        be None.
    size : int, optional
        Size of device memory in bytes. Must be specified if `data` is an
        integer.
    owner : object, optional
        Python object to which the lifetime of the memory allocation is tied.
        A reference to this object is kept in the returned Buffer.
    """

    _ptr: int
    _size: int
    _owner: object

    def __init__(
        self, data: Union[int, Any], *, size: int = None, owner: object = None
    ):
        if isinstance(data, int):
            if size is None:
                raise ValueError(
                    "size must be specified when `data` is an integer"
                )
            if size < 0:
                raise ValueError("size cannot be negative")
            self._ptr = data
            self._size = size
            self._owner = owner
            return
        if size is not None or owner is not None:
            raise ValueError(
                "`size` and `owner` must be None when "
                "`data` is a buffer-like object"
            )
        # `data` is a buffer-like or array-like object
        buf: Any = data
        if isinstance(buf, rmm.DeviceBuffer):
            self._ptr = buf.ptr
            self._size = buf.size
            self._owner = buf
            return
        iface = getattr(buf, "__cuda_array_interface__", None)
        if iface:
            self._ptr, self._size = get_ptr_and_size(iface)
            self._owner = buf
            return
        # At this point, `buf` must represents host memory, let's deligate
        # to `._init_from_host_memory()`
        buf = memoryview(buf)
        if not buf.c_contiguous:
            raise ValueError("`data` must be C-contiguous")
        self._init_from_host_memory(buf)

    def _init_from_host_memory(self, data: memoryview) -> None:
        """Initialize in the case where `data` represents host memory.

        Sub-classes can overwrite this and still use `super().__init__()`
        to handle the (trivial) case where `data` represents device memory.

        This default implemention copies `data` to a newly allocated RMM
        device buffer.
        """
        ptr, size = get_ptr_and_size(numpy.asarray(data).__array_interface__)
        buf = rmm.DeviceBuffer(ptr=ptr, size=size)
        self._ptr = buf.ptr
        self._size = buf.size
        self._owner = buf

    @classmethod
    def from_device_memory(cls, data: Any) -> Buffer:
        ret = cls.__new__(cls)
        if isinstance(data, rmm.DeviceBuffer):
            ret._ptr = data.ptr
            ret._size = data.size
            ret._owner = data
        else:
            ret._ptr, ret._size = get_ptr_and_size(
                data.__cuda_array_interface__
            )
            ret._owner = data
        return ret

    @classmethod
    def from_host_memory(cls, data: Any) -> Buffer:
        ret = cls.__new__(cls)
        ptr, size = get_ptr_and_size(numpy.asarray(data).__array_interface__)
        buf = rmm.DeviceBuffer(ptr=ptr, size=size)
        ret._ptr = buf.ptr
        ret._size = buf.size
        ret._owner = buf
        return ret

    def _getitem(self, offset: int, size: int) -> Buffer:
        """
        Sub-classes can overwrite this to implement __getitem__
        without having to handle non-slice inputs.
        """
        return self.__class__(
            data=self.ptr + offset, size=size, owner=self.owner
        )

    def __getitem__(self, key: slice) -> Buffer:
        if not isinstance(key, slice):
            raise TypeError(
                "Argument 'key' has incorrect type "
                f"(expected slice, got {key.__class__.__name__})"
            )
        start, stop, step = key.indices(self.size)
        if step != 1:
            raise ValueError("slice must be C-contiguous")
        return self._getitem(offset=start, size=stop - start)

    @property
    def size(self) -> int:
        return self._size

    @property
    def nbytes(self) -> int:
        return self._size

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def owner(self) -> Any:
        return self._owner

    @property
    def __cuda_array_interface__(self) -> dict:
        return {
            "data": (self.ptr, False),
            "shape": (self.size,),
            "strides": None,
            "typestr": "|u1",
            "version": 0,
        }

    def memoryview(self) -> memoryview:
        host_buf = bytearray(self.size)
        rmm._lib.device_buffer.copy_ptr_to_host(self.ptr, host_buf)
        return memoryview(host_buf).toreadonly()

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
        if header["frame_count"] != 1:
            raise ValueError("Deserializing a Buffer expect a single frame")
        frame = frames[0]
        if isinstance(frame, cls):
            return frame  # The frame is already deserialized
        return cls(frame)

    def __repr__(self) -> str:
        return (
            f"<cudf.core.buffer.Buffer size={format_bytes(self._size)} "
            f"ptr={hex(self._ptr)} owner={repr(self._owner)} "
        )


def is_c_contiguous(
    shape: Sequence[int], strides: Sequence[int], itemsize: int
) -> bool:
    """
    Determine if shape and strides are C-contiguous

    Parameters
    ----------
    shape : Sequence[int]
        Number of elements in each dimension.
    strides : Sequence[int]
        The stride of each dimension in bytes.
    itemsize : int
        Size of an element in bytes.

    Return
    ------
    bool
        The boolean answer.
    """

    if any(dim == 0 for dim in shape):
        return True
    cumulative_stride = itemsize
    for dim, stride in zip(reversed(shape), reversed(strides)):
        if dim > 1 and stride != cumulative_stride:
            return False
        cumulative_stride *= dim
    return True


def get_ptr_and_size(array_interface: Mapping) -> Tuple[int, int]:
    """
    Retrieve the pointer and size from an array interface.

    Raises ValueError if array isn't C-contiguous.

    Parameters
    ----------
    array_interface : Mapping
        The array interface metadata.

    Return
    ------
    pointer : int
        The pointer to device or host memory
    size : int
        The size in bytes
    """

    shape = array_interface["shape"] or (1,)
    strides = array_interface["strides"]
    itemsize = cudf.dtype(array_interface["typestr"]).itemsize
    if strides is None or is_c_contiguous(shape, strides, itemsize):
        nelem = math.prod(shape)
        ptr = array_interface["data"][0] or 0
        return ptr, nelem * itemsize
    raise ValueError("Buffer data must be C-contiguous")
