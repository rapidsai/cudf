# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from __future__ import annotations

import math
import pickle
from types import SimpleNamespace
from typing import (
    Any,
    Dict,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

import numpy
from typing_extensions import Self

import rmm

import cudf
from cudf.core.abc import Serializable
from cudf.utils.string import format_bytes

T = TypeVar("T")


def get_owner(data, klass: Type[T]) -> Optional[T]:
    """Get the owner of `data`, if any exist

    Search through the stack of data owners in order to find an
    owner of type `klass` (not subclasses).

    Parameters
    ----------
    data
        The data object

    Return
    ------
    klass or None
        The owner of `data` if `klass` or None.
    """

    if type(data) is klass:
        return data
    if hasattr(data, "owner"):
        return get_owner(data.owner, klass)
    return None


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


class BufferOwner(Serializable):
    """A owning buffer that represents device memory.

    This class isn't meant to be used throughout cuDF. Instead, it
    standardizes data owning by wrapping any data object that
    represents device memory. Multiple `Buffer` instances, which are
    the ones used throughout cuDF, can then refer to the same
    `BufferOwner` instance.

    Use `_from_device_memory` and `_from_host_memory` to create
    a new instance from either device or host memory respectively.
    """

    _ptr: int
    _size: int
    _owner: object

    @classmethod
    def _from_device_memory(cls, data: Any) -> Self:
        """Create from an object exposing `__cuda_array_interface__`.

        No data is being copied.

        Parameters
        ----------
        data : device-buffer-like
            An object implementing the CUDA Array Interface.

        Returns
        -------
        BufferOwner
            BufferOwner wrapping `data`
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
    def _from_host_memory(cls, data: Any) -> Self:
        """Create an owner from a buffer or array like object

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
        BufferOwner
            BufferOwner wrapping a device copy of `data`.
        """

        # Convert to numpy array, this will not copy data in most cases.
        ary = numpy.array(data, copy=False, subok=True)
        # Extract pointer and size
        ptr, size = get_ptr_and_size(ary.__array_interface__)
        # Copy to device memory
        buf = rmm.DeviceBuffer(ptr=ptr, size=size)
        # Create from device memory
        return cls._from_device_memory(buf)

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
        return {
            "data": (self.get_ptr(mode="write"), False),
            "shape": (self.size,),
            "strides": None,
            "typestr": "|u1",
            "version": 0,
        }

    def get_ptr(self, *, mode: Literal["read", "write"]) -> int:
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

        Returns
        -------
        int
            The device pointer as an integer

        See Also
        --------
        SpillableBuffer.get_ptr
        ExposureTrackedBuffer.get_ptr
        """
        return self._ptr

    def memoryview(
        self, *, offset: int = 0, size: Optional[int] = None
    ) -> memoryview:
        """Read-only access to the buffer through host memory."""
        size = self._size if size is None else size
        host_buf = host_memory_allocation(size)
        rmm._lib.device_buffer.copy_ptr_to_host(
            self.get_ptr(mode="read") + offset, host_buf
        )
        return memoryview(host_buf).toreadonly()

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} size={format_bytes(self._size)} "
            f"ptr={hex(self._ptr)} owner={repr(self._owner)}>"
        )


class Buffer(Serializable):
    """A buffer that represents a slice or view of a `BufferOwner`.

    Use the factory function `as_buffer` to create a Buffer instance.
    """

    def __init__(
        self,
        *,
        owner: BufferOwner,
        offset: int,
        size: int,
    ) -> None:
        if size < 0:
            raise ValueError("size cannot be negative")
        if offset < 0:
            raise ValueError("offset cannot be negative")
        if offset + size > owner.size:
            raise ValueError(
                "offset+size cannot be greater than the size of owner"
            )
        self._owner = owner
        self._offset = offset
        self._size = size

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

    def __getitem__(self, key: slice) -> Self:
        """Create a new slice of the buffer."""
        if not isinstance(key, slice):
            raise TypeError(
                "Argument 'key' has incorrect type "
                f"(expected slice, got {key.__class__.__name__})"
            )
        start, stop, step = key.indices(self.size)
        if step != 1:
            raise ValueError("slice must be C-contiguous")
        return self.__class__(
            owner=self._owner, offset=self._offset + start, size=stop - start
        )

    def get_ptr(self, *, mode: Literal["read", "write"]) -> int:
        return self._owner.get_ptr(mode=mode) + self._offset

    def memoryview(
        self, *, offset: int = 0, size: Optional[int] = None
    ) -> memoryview:
        size = self._size if size is None else size
        return self._owner.memoryview(offset=self._offset + offset, size=size)

    def copy(self, deep: bool = True) -> Self:
        """Return a copy of Buffer.

        Parameters
        ----------
        deep : bool, default True
            - If deep=True, returns a deep copy of the underlying data.
            - If deep=False, returns a new `Buffer` instance that refers
              to the same `BufferOwner` as this one. Thus, no device
              data are being copied.

        Returns
        -------
        Buffer
            A new buffer that either refers to either a new or an existing
            `BufferOwner` depending on the `deep` argument (see above).
        """

        # When doing a shallow copy, we just return a new slice
        if not deep:
            return self.__class__(
                owner=self._owner, offset=self._offset, size=self._size
            )

        # Otherwise, we create a new copy of the memory
        owner = self._owner._from_device_memory(
            rmm.DeviceBuffer(
                ptr=self._owner.get_ptr(mode="read") + self._offset,
                size=self.size,
            )
        )
        return self.__class__(owner=owner, offset=0, size=owner.size)

    @property
    def __cuda_array_interface__(self) -> Mapping:
        """Implementation of the CUDA Array Interface."""
        return {
            "data": (self.get_ptr(mode="write"), False),
            "shape": (self.size,),
            "strides": None,
            "typestr": "|u1",
            "version": 0,
        }

    def serialize(self) -> Tuple[dict, list]:
        """Serialize the buffer into header and frames.

        The frames can be a mixture of memoryview, Buffer, and BufferOwner
        objects.

        Returns
        -------
        Tuple[dict, List]
            The first element of the returned tuple is a dict containing any
            serializable metadata required to reconstruct the object. The
            second element is a list containing single frame.
        """
        header: Dict[str, Any] = {}
        header["type-serialized"] = pickle.dumps(type(self))
        header["owner-type-serialized"] = pickle.dumps(type(self._owner))
        header["frame_count"] = 1
        frames = [self]
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> Self:
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

        owner_type: BufferOwner = pickle.loads(header["owner-type-serialized"])
        if hasattr(frame, "__cuda_array_interface__"):
            owner = owner_type._from_device_memory(frame)
        else:
            owner = owner_type._from_host_memory(frame)
        return cls(
            owner=owner,
            offset=0,
            size=owner.size,
        )

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} size={format_bytes(self._size)} "
            f"offset={format_bytes(self._offset)} of {self._owner}>"
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
