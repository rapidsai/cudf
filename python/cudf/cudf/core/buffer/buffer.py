# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from __future__ import annotations

import math
import pickle
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type, TypeVar

import numpy

import rmm

import cudf
from cudf.core.abc import Serializable
from cudf.utils.string import format_bytes

T = TypeVar("T", bound="Buffer")


def host_memory_allocation(nbytes: int) -> memoryview:
    """Allocate host memory using NumPy

    This is an alternative to `bytearray` to avoid memory initialization cost.
    A `bytearray` is zero-initialized using `calloc`, which we don't need.
    Additionally, `numpy.empty` both skips the zero-initialization and uses
    hugepages when available <https://github.com/numpy/numpy/pull/14216>.

    Parameters
    ----------
    nbytes : int
        Size of the new host allocation in bytes.

    Return
    ------
    memoryview
        The new host allocation.
    """
    return numpy.empty((nbytes,), dtype="u1").data


def cuda_array_interface_wrapper(
    ptr: int,
    size: int,
    owner: Optional[object] = None,
    readonly=False,
    typestr="|u1",
    version=0,
):
    """Wrap device pointer in an object that exposes `__cuda_array_interface__`

    See <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>

    Parameters
    ----------
    ptr : int
        An integer representing a pointer to device memory.
    size : int, optional
        Size of device memory in bytes.
    owner : object, optional
        Python object to which the lifetime of the memory allocation is tied.
        A reference to this object is kept in the returned wrapper object.
    readonly: bool, optional
        Mark the interface read-only.
    typestr: str, optional
        The type string of the interface. By default this is "|u1", which
        means "an unsigned integer with a not relevant byteorder". See:
        <https://numpy.org/doc/stable/reference/arrays.interface.html>
    version : bool, optional
        The version of the interface.

    Return
    ------
    SimpleNamespace
        An object that exposes `__cuda_array_interface__` and keeps a reference
        to `owner`.
    """

    if size < 0:
        raise ValueError("size cannot be negative")

    return SimpleNamespace(
        __cuda_array_interface__={
            "data": (ptr, readonly),
            "shape": (size,),
            "strides": None,
            "typestr": typestr,
            "version": version,
        },
        owner=owner,
    )


class Buffer(Serializable):
    """A Buffer represents device memory.

    Use the factory function `as_buffer` to create a Buffer instance.
    """

    _ptr: int
    _size: int
    _owner: object

    def __init__(self):
        raise ValueError(
            f"do not create a {self.__class__} directly, please "
            "use the factory function `cudf.core.buffer.as_buffer`"
        )

    @classmethod
    def _from_device_memory(cls: Type[T], data: Any) -> T:
        """Create a Buffer from an object exposing `__cuda_array_interface__`.

        No data is being copied.

        Parameters
        ----------
        data : device-buffer-like
            An object implementing the CUDA Array Interface.

        Returns
        -------
        Buffer
            Buffer representing the same device memory as `data`
        """

        # Bypass `__init__` and initialize attributes manually
        ret = cls.__new__(cls)
        ret._owner = data
        if isinstance(data, rmm.DeviceBuffer):  # Common case shortcut
            ret._ptr = data.ptr
            ret._size = data.size
        else:
            ret._ptr, ret._size = get_ptr_and_size(
                data.__cuda_array_interface__
            )
        if ret.size < 0:
            raise ValueError("size cannot be negative")
        return ret

    @classmethod
    def _from_host_memory(cls: Type[T], data: Any) -> T:
        """Create a Buffer from a buffer or array like object

        Data must implement `__array_interface__`, the buffer protocol, and/or
        be convertible to a buffer object using `numpy.array()`

        The host memory is copied to a new device allocation.

        Raises ValueError if array isn't C-contiguous.

        Parameters
        ----------
        data : Any
            An object that represens host memory.

        Returns
        -------
        Buffer
            Buffer representing a copy of `data`.
        """

        # Convert to numpy array, this will not copy data in most cases.
        ary = numpy.array(data, copy=False, subok=True)
        # Extract pointer and size
        ptr, size = get_ptr_and_size(ary.__array_interface__)
        # Copy to device memory
        buf = rmm.DeviceBuffer(ptr=ptr, size=size)
        # Create from device memory
        return cls._from_device_memory(buf)

    def _getitem(self, offset: int, size: int) -> Buffer:
        """
        Sub-classes can overwrite this to implement __getitem__
        without having to handle non-slice inputs.
        """
        return self._from_device_memory(
            cuda_array_interface_wrapper(
                ptr=self.get_ptr(mode="read") + offset,
                size=size,
                owner=self.owner,
            )
        )

    def __getitem__(self, key: slice) -> Buffer:
        """Create a new slice of the buffer."""
        if not isinstance(key, slice):
            raise TypeError(
                "Argument 'key' has incorrect type "
                f"(expected slice, got {key.__class__.__name__})"
            )
        start, stop, step = key.indices(self.size)
        if step != 1:
            raise ValueError("slice must be C-contiguous")
        return self._getitem(offset=start, size=stop - start)

    def copy(self, deep: bool = True):
        """
        Return a copy of Buffer.

        Parameters
        ----------
        deep : bool, default True
            If True, returns a deep copy of the underlying Buffer data.
            If False, returns a shallow copy of the Buffer pointing to
            the same underlying data.

        Returns
        -------
        Buffer
        """
        if deep:
            return self._from_device_memory(
                rmm.DeviceBuffer(ptr=self.get_ptr(mode="read"), size=self.size)
            )
        else:
            return self[:]

    @property
    def size(self) -> int:
        """Size of the buffer in bytes."""
        return self._size

    @property
    def nbytes(self) -> int:
        """Size of the buffer in bytes."""
        return self._size

    @property
    def owner(self) -> Any:
        """Object owning the memory of the buffer."""
        return self._owner

    @property
    def __cuda_array_interface__(self) -> Mapping:
        """Implementation of the CUDA Array Interface."""
        return self._get_cuda_array_interface(readonly=False)

    def _get_cuda_array_interface(self, readonly=False):
        """Helper function to create a CUDA Array Interface.

        Parameters
        ----------
        readonly : bool, default False
            If True, returns a CUDA Array Interface with
            readonly flag set to True.
            If False, returns a CUDA Array Interface with
            readonly flag set to False.

        Returns
        -------
        dict
        """
        return {
            "data": (
                self.get_ptr(mode="read" if readonly else "write"),
                readonly,
            ),
            "shape": (self.size,),
            "strides": None,
            "typestr": "|u1",
            "version": 0,
        }

    def get_ptr(self, *, mode) -> int:
        """Device pointer to the start of the buffer.

        Parameters
        ----------
        mode : str
            Supported values are {"read", "write"}
            If "write", the data pointed to may be modified
            by the caller. If "read", the data pointed to
            must not be modified by the caller.
            Failure to fulfill this contract will cause
            incorrect behavior.


        See Also
        --------
        SpillableBuffer.get_ptr
        CopyOnWriteBuffer.get_ptr
        """
        return self._ptr

    def memoryview(self) -> memoryview:
        """Read-only access to the buffer through host memory."""
        host_buf = host_memory_allocation(self.size)
        rmm._lib.device_buffer.copy_ptr_to_host(
            self.get_ptr(mode="read"), host_buf
        )
        return memoryview(host_buf).toreadonly()

    def serialize(self) -> Tuple[dict, list]:
        """Serialize the buffer into header and frames.

        The frames can be a mixture of memoryview and Buffer objects.

        Returns
        -------
        Tuple[dict, List]
            The first element of the returned tuple is a dict containing any
            serializable metadata required to reconstruct the object. The
            second element is a list containing Buffers and memoryviews.
        """
        header: Dict[str, Any] = {}
        header["type-serialized"] = pickle.dumps(type(self))
        header["frame_count"] = 1
        frames = [self]
        return header, frames

    @classmethod
    def deserialize(cls: Type[T], header: dict, frames: list) -> T:
        """Create an Buffer from a serialized representation.

        Parameters
        ----------
        header : dict
            The metadata required to reconstruct the object.
        frames : list
            The Buffer and memoryview that makes up the Buffer.

        Returns
        -------
        Buffer
            The deserialized Buffer.
        """
        if header["frame_count"] != 1:
            raise ValueError("Deserializing a Buffer expect a single frame")
        frame = frames[0]
        if isinstance(frame, cls):
            return frame  # The frame is already deserialized

        if hasattr(frame, "__cuda_array_interface__"):
            return cls._from_device_memory(frame)
        return cls._from_host_memory(frame)

    def __repr__(self) -> str:
        klass = self.__class__
        name = f"{klass.__module__}.{klass.__qualname__}"
        return (
            f"<{name} size={format_bytes(self._size)} "
            f"ptr={hex(self._ptr)} owner={repr(self._owner)}>"
        )


def is_c_contiguous(
    shape: Sequence[int], strides: Sequence[int], itemsize: int
) -> bool:
    """Determine if shape and strides are C-contiguous

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
    """Retrieve the pointer and size from an array interface.

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
