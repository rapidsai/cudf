# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from __future__ import annotations

import math
import pickle
import warnings
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np

import rmm

import cudf
from cudf.core.abc import Serializable
from cudf.utils.string import format_bytes

# Frame type for serialization and deserialization of `DeviceBufferLike`
Frame = Union[memoryview, "DeviceBufferLike"]


class _ProtocolMeta(Protocol.__class__):  # type: ignore
    """Custom implementation of @runtime_checkable

    The native implementation of @runtime_checkable use `hasattr()`
    to check if an instance implements the Protocol. This is a problem
    when checking SpillableBuffer.ptr because it will unspill the
    buffer and make it unspillable permanently.
    """

    def __instancecheck__(cls, instance):
        if isinstance(instance, Buffer):
            return True
        return super().__instancecheck__(instance)


@runtime_checkable
class DeviceBufferLike(Protocol, metaclass=_ProtocolMeta):
    def __getitem__(self, key: slice) -> DeviceBufferLike:
        """Create a new view of the buffer."""

    @property
    def size(self) -> int:
        """Size of the buffer in bytes."""

    @property
    def nbytes(self) -> int:
        """Size of the buffer in bytes."""

    @property
    def ptr(self) -> int:
        """Device pointer to the start of the buffer."""

    @property
    def owner(self) -> Any:
        """Object owning the memory of the buffer."""

    @property
    def __cuda_array_interface__(self) -> Mapping:
        """Implementation of the CUDA Array Interface."""

    def memoryview(self) -> memoryview:
        """Read-only access to the buffer through host memory."""

    def serialize(self) -> Tuple[dict, List[Frame]]:
        """Serialize the buffer into header and frames.

        The frames can be a mixture of memoryview and device-buffer-like
        objects.

        Returns
        -------
        Tuple[Dict, List]
            The first element of the returned tuple is a dict containing any
            serializable metadata required to reconstruct the object. The
            second element is a list containing the device buffers and
            memoryviews of the object.
        """

    @classmethod
    def deserialize(
        cls, header: dict, frames: List[Frame]
    ) -> DeviceBufferLike:
        """Generate an buffer from a serialized representation.

        Parameters
        ----------
        header : dict
            The metadata required to reconstruct the object.
        frames : list
            The device-buffer-like and memoryview buffers that the object
            should contain.

        Returns
        -------
        DeviceBufferLike
            A new object that implements DeviceBufferLike.
        """


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
            warnings.warn(
                "Creating a Buffer() from an integer is deprecated and will "
                "raise in a future version, use as_device_buffer_like() "
                "instead.",
                FutureWarning,
            )
            self._ptr = data
            self._size = size
            self._owner = owner
        else:
            if size is not None or owner is not None:
                raise ValueError(
                    "`size` and `owner` must be None when "
                    "`data` is a buffer-like object"
                )

            # `data` is a buffer-like object
            buf: Any = data
            if isinstance(buf, rmm.DeviceBuffer):
                self._ptr = buf.ptr
                self._size = buf.size
            else:
                iface = getattr(buf, "__cuda_array_interface__", None)
                if iface:
                    ptr, size = get_ptr_and_size(iface)
                    self._ptr = ptr
                    self._size = size
                else:
                    ptr, size = get_ptr_and_size(
                        np.asarray(buf).__array_interface__
                    )
                    buf = rmm.DeviceBuffer(ptr=ptr, size=size)
                    self._ptr = buf.ptr
                    self._size = buf.size
            self._owner = buf

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
