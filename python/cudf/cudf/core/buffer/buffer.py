# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from __future__ import annotations

import math
import pickle
import weakref
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import numpy
from typing_extensions import Self

import pylibcudf
import rmm

import cudf
from cudf.core.abc import Serializable
from cudf.utils.string import format_bytes

if TYPE_CHECKING:
    from collections.abc import Mapping


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
    owner: object | None = None,
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
    """An owning buffer that represents device memory.

    This class isn't meant to be used throughout cuDF. Instead, it
    standardizes data owning by wrapping any data object that
    represents device memory. Multiple `Buffer` instances, which are
    the ones used throughout cuDF, can then refer to the same
    `BufferOwner` instance.

    In order to implement copy-on-write and spillable buffers, we need the
    ability to detect external access to the underlying memory. We say that
    the buffer has been exposed if the device pointer (integer or void*) has
    been accessed outside of BufferOwner. In this case, we have no control
    over knowing if the data is being modified by a third party.

    Use `from_device_memory` and `from_host_memory` to create
    a new instance from either device or host memory respectively.

    Parameters
    ----------
    ptr
        An integer representing a pointer to memory.
    size
        The size of the memory in nbytes
    owner
        Python object to which the lifetime of the memory allocation is tied.
        This buffer will keep a reference to `owner`.
    exposed
        Pointer to the underlying memory

    Raises
    ------
    ValueError
        If size is negative
    """

    _ptr: int
    _size: int
    _owner: object
    _exposed: bool
    # The set of buffers that point to this owner.
    _slices: weakref.WeakSet[Buffer]

    def __init__(
        self,
        *,
        ptr: int,
        size: int,
        owner: object,
        exposed: bool,
    ):
        if size < 0:
            raise ValueError("size cannot be negative")

        self._ptr = ptr
        self._size = size
        self._owner = owner
        self._exposed = exposed
        self._slices = weakref.WeakSet()

    @classmethod
    def from_device_memory(cls, data: Any, exposed: bool) -> Self:
        """Create from an object providing a `__cuda_array_interface__`.

        No data is being copied.

        Parameters
        ----------
        data : device-buffer-like
            An object implementing the CUDA Array Interface.
        exposed : bool
            Mark the buffer as permanently exposed. This is used by
            ExposureTrackedBuffer to determine when a deep copy is required
            and by SpillableBuffer to mark the buffer unspillable.

        Returns
        -------
        BufferOwner
            BufferOwner wrapping `data`

        Raises
        ------
        AttributeError
            If data does not support the cuda array interface
        ValueError
            If the resulting buffer has negative size
        """

        if isinstance(data, rmm.DeviceBuffer):  # Common case shortcut
            ptr = data.ptr
            size = data.size
        else:
            ptr, size = get_ptr_and_size(data.__cuda_array_interface__)
        return cls(ptr=ptr, size=size, owner=data, exposed=exposed)

    @classmethod
    def from_host_memory(cls, data: Any) -> Self:
        """Create an owner from a buffer or array like object

        Data must implement `__array_interface__`, the buffer protocol, and/or
        be convertible to a buffer object using `numpy.asanyarray()`

        The host memory is copied to a new device allocation.

        Raises ValueError if array isn't C-contiguous.

        Parameters
        ----------
        data : Any
            An object that represents host memory.

        Returns
        -------
        BufferOwner
            BufferOwner wrapping a device copy of `data`.
        """

        # Convert to numpy array, this will not copy data in most cases.
        ary = numpy.asanyarray(data)
        # Extract pointer and size
        ptr, size = get_ptr_and_size(ary.__array_interface__)
        # Copy to device memory
        buf = rmm.DeviceBuffer(ptr=ptr, size=size)
        # Create from device memory
        return cls.from_device_memory(buf, exposed=False)

    @property
    def size(self) -> int:
        """Size of the buffer in bytes."""
        return self._size

    @property
    def nbytes(self) -> int:
        """Size of the buffer in bytes."""
        return self._size

    @property
    def owner(self) -> object:
        """Object owning the memory of the buffer."""
        return self._owner

    @property
    def exposed(self) -> bool:
        """The current exposure status of the buffer

        This is used by ExposureTrackedBuffer to determine when a deep copy
        is required and by SpillableBuffer to mark the buffer unspillable.
        """
        return self._exposed

    def mark_exposed(self) -> None:
        """Mark the buffer as "exposed" permanently

        This is used by ExposureTrackedBuffer to determine when a deep copy
        is required and by SpillableBuffer to mark the buffer unspillable.

        Notice, once the exposure status becomes True, it will never change
        back.
        """
        self._exposed = True

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
        self, *, offset: int = 0, size: int | None = None
    ) -> memoryview:
        """Read-only access to the buffer through host memory."""
        size = self._size if size is None else size
        host_buf = host_memory_allocation(size)
        rmm.pylibrmm.device_buffer.copy_ptr_to_host(
            self.get_ptr(mode="read") + offset, host_buf
        )
        return memoryview(host_buf).toreadonly()

    def __str__(self) -> str:
        return (
            f"<{self.__class__.__name__} size={format_bytes(self._size)} "
            f"ptr={hex(self._ptr)} owner={self._owner!r}>"
        )


class Buffer(Serializable):
    """A buffer that represents a slice or view of a `BufferOwner`.

    Use the factory function `as_buffer` to create a Buffer instance.

    Note
    ----
    This buffer is untyped, so all indexing and sizes are in bytes.

    Parameters
    ----------
    owner
        The owning exposure buffer this refers to.
    offset
        The offset relative to the start memory of owner (in bytes).
    size
        The size of the buffer (in bytes). If None, use the size of owner.
    """

    def __init__(
        self,
        *,
        owner: BufferOwner,
        offset: int = 0,
        size: int | None = None,
    ) -> None:
        size = owner.size if size is None else size
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
    def owner(self) -> BufferOwner:
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

    def memoryview(self) -> memoryview:
        return self._owner.memoryview(offset=self._offset, size=self._size)

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
        owner = self._owner.from_device_memory(
            rmm.DeviceBuffer(
                ptr=self._owner.get_ptr(mode="read") + self._offset,
                size=self.size,
            ),
            exposed=False,
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

    def serialize(self) -> tuple[dict, list]:
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
        header: dict[str, Any] = {}
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
            owner = owner_type.from_device_memory(frame, exposed=False)
        else:
            owner = owner_type.from_host_memory(frame)
        return cls(
            owner=owner,
            offset=0,
            size=owner.size,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(owner={self._owner!r}, "
            f"offset={self._offset!r}, size={self._size!r})"
        )

    def __str__(self) -> str:
        return (
            f"<{self.__class__.__name__} size={format_bytes(self._size)} "
            f"offset={format_bytes(self._offset)} of {self._owner}>"
        )


def get_ptr_and_size(array_interface: Mapping) -> tuple[int, int]:
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
    if strides is None or pylibcudf.column.is_c_contiguous(
        shape, strides, itemsize
    ):
        nelem = math.prod(shape)
        ptr = array_interface["data"][0] or 0
        return ptr, nelem * itemsize
    raise ValueError("Buffer data must be C-contiguous")
