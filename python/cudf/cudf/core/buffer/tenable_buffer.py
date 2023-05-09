# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from __future__ import annotations

import weakref
from typing import Any, Container, Mapping, Optional, Type, TypeVar, cast

import cudf
from cudf.core.buffer.buffer import Buffer, get_ptr_and_size
from cudf.utils.string import format_bytes

T = TypeVar("T", bound="TenableBuffer")

# Alternativ names:
#  - ShieldedBuffer
#  - TenableBuffer
#  - ExposeTrackedBuffer
#  - UniqueBuffer
#  - IsolatedBuffer
#  - IsolatableBuffer


def get_tenable_owner(data) -> Optional[TenableBuffer]:
    """Get the tenable owner of `data`, if any exist

    Search through the stack of data owners in order to find an
    owner of type `TenableBuffer` (not subclasses).

    Parameters
    ----------
    data : buffer-like or array-like
        A buffer-like or array-like object that represent C-contiguous memory.

    Return
    ------
    TenableBuffer or None
        The owner of `data` if TenableBuffer or None.
    """

    if type(data) is TenableBuffer:
        return data
    if hasattr(data, "owner"):
        return get_tenable_owner(data.owner)
    return None


def as_tenable_buffer(data, exposed: bool) -> TenableBuffer:
    if not hasattr(data, "__cuda_array_interface__"):
        if exposed:
            raise ValueError("cannot created exposed host memory")
        return TenableBuffer._from_host_memory(data)[:]

    tenable_owner = get_tenable_owner(data)
    if tenable_owner is None:
        return TenableBuffer._from_device_memory(data, exposed=exposed)[:]

    # At this point, we know that `data` is owned by a tenable buffer
    ptr, size = get_ptr_and_size(data.__cuda_array_interface__)
    base_ptr = tenable_owner.get_raw_ptr()
    return BufferSlice(base=tenable_owner, offset=ptr - base_ptr, size=size)


class TenableBuffer(Buffer):
    """A Buffer that tracks its "expose" status.

    In order to implement copy-on-write and spillable buffers, we need the
    ability to detect external access to the underlying memory. We say that
    the buffer has been exposed if the device pointer (integer or void*) has
    been accessed outside of TenableBuffer. In this case, we have no control
    over knowing if the data is being modified by a third-party.

    Attributes
    ----------
    _exposed
        The current expose status of the buffer. Notice, once the expose status
        becomes False, it should never change back.
    _slices
        The set of BufferSlice that points to this buffer.
    """

    _exposed: bool
    _slices: weakref.WeakSet[BufferSlice]

    @property
    def exposed(self) -> bool:
        return self._exposed

    def mark_exposed(self) -> None:
        """Mark the buffer as "exposed" permanently"""
        self._exposed = True

    @classmethod
    def _from_device_memory(
        cls: Type[T], data: Any, *, exposed: bool = False
    ) -> T:
        """Create an tenable buffer from device memory.

        No data is being copied.

        Parameters
        ----------
        data : device-buffer-like
            An object implementing the CUDA Array Interface.
        exposed : bool, optional
            Mark the buffer as permanently exposed.

        Returns
        -------
        TenableBuffer
            Buffer representing the same device memory as `data`
        """
        ret = super()._from_device_memory(data)
        ret._exposed = exposed
        ret._slices = weakref.WeakSet()
        return ret

    def _getitem(self, offset: int, size: int) -> BufferSlice:
        return BufferSlice(base=self, offset=offset, size=size)

    def get_raw_ptr(self) -> int:
        """Get the memory pointer this buffer.

        Warning, this memory pointer can mean many things!
        Warning, it is not safe to access the pointer value without
        spill lock the buffer manually.

        This method neither exposes nor spill locks the buffer.

        Return
        ------
        int
            The memory pointer as an integer (device or host memory)
        """
        return self._ptr

    @property
    def __cuda_array_interface__(self) -> Mapping:
        self.mark_exposed()
        return super().__cuda_array_interface__

    def _get_cuda_array_interface(self, readonly=False):
        # TODO: remove, use `self.get_raw_ptr()` directly instead
        return {
            "data": (self.get_raw_ptr(), readonly),
            "shape": (self.size,),
            "strides": None,
            "typestr": "|u1",
            "version": 0,
        }


class BufferSlice(TenableBuffer):
    """A slice (aka. a view) of a tenable buffer.

    Parameters
    ----------
    base
        The tenable buffer this slice refers to.
    offset
        The offset relative to the start memory of base (in bytes).
    size
        The size of the slice (in bytes)
    passthrough_attributes
        Attribute names that are passed through to the base as-is.
    """

    def __init__(
        self,
        base: TenableBuffer,
        offset: int,
        size: int,
        *,
        passthrough_attributes: Container[str] = ("exposed",),
    ) -> None:
        if size < 0:
            raise ValueError("size cannot be negative")
        if offset < 0:
            raise ValueError("offset cannot be negative")
        if offset + size > base.size:
            raise ValueError(
                "offset+size cannot be greater than the size of base"
            )
        self._base = base
        self._offset = offset
        self._size = size
        self._owner = base
        self._passthrough_attributes = passthrough_attributes
        base._slices.add(self)

    def __getattr__(self, name):
        if name in self._passthrough_attributes:
            return getattr(self._base, name)
        raise AttributeError(
            f"{self.__class__.__name__} object has no attribute {name}"
        )

    def _getitem(self, offset: int, size: int) -> BufferSlice:
        return BufferSlice(
            base=self._base, offset=offset + self._offset, size=size
        )

    @property
    def exposed(self) -> bool:
        return self._base.exposed

    def get_ptr(self, mode: str = "write") -> int:
        if mode == "write" and cudf.get_option("copy_on_write"):
            self.make_single_owner_inplace()
        return self._base.get_ptr(mode=mode) + self._offset

    def get_raw_ptr(self) -> int:
        return self._base.get_raw_ptr() + self._offset

    def memoryview(
        self, *, offset: int = 0, size: Optional[int] = None
    ) -> memoryview:
        return self._base.memoryview(offset=self._offset + offset, size=size)

    def copy(self, deep: bool = True) -> BufferSlice:
        """Return a copy of Buffer.

        What actually happens when `deep == False` is altered by the
        "copy_on_write" option. When copy-on-write is enabled, a shallow copy
        because a deep copy if the buffer has been exposed. This is because we
        have no control over knowing if the data is being modified when the
        buffer has been exposed to third-party.

        Parameters
        ----------
        deep : bool, default True
            The meaning when copy-on-write is disabled:
                - If True, returns a deep copy of the underlying Buffer data.
                - If False, returns a shallow copy of the Buffer pointing to
                  the same underlying data.
            The meaning when copy-on-write is enabled:
                - Always a deep copy of the underlying Buffer data.

        Returns
        -------
        Buffer
        """
        if deep or not cudf.get_option("copy_on_write"):
            base_copy = self._base.copy(deep=deep)
        else:
            base_copy = self._base.copy(deep=self.exposed)
        return cast(
            BufferSlice, base_copy[self._offset : self._offset + self._size]
        )

    @property
    def __cuda_array_interface__(self) -> Mapping:
        if cudf.get_option("copy_on_write"):
            self.make_single_owner_inplace()
        return super().__cuda_array_interface__

    def make_single_owner_inplace(self) -> None:
        """Make sure this slice is the only own pointing to `._base`

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

        if len(self._base._slices) > 1:
            t = self.copy(deep=True)
            self._base = t._base
            self._offset = t._offset
            self._size = t._size
            self._owner = t._base
            t._base._slices.add(self)

    def __repr__(self) -> str:
        return (
            f"<BufferSlice size={format_bytes(self._size)} "
            f"offset={format_bytes(self._offset)} of {self._base} "
        )
