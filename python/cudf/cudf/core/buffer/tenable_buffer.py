# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from __future__ import annotations

import weakref
from typing import Any, Container, Mapping, Optional, Type, TypeVar, cast

import cudf
from cudf.core.buffer.buffer import Buffer, get_ptr_and_size
from cudf.utils.string import format_bytes

T = TypeVar("T", bound="TenableBuffer")


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


def as_tenable_buffer(
    data, exposed: bool, subclass: Optional[Type[T]] = None
) -> BufferSlice:
    """Factory function to wrap `data` in a tenable buffer and buffer slice

    If `subclass` is None, a new TenableBuffer that points to the memory of
    `data` is created and a BufferSlice that points to all of the new
    TenableBuffer is returned.

    If `subclass` is not None, a new `subclass` is created instead. Still,
    a BufferSlice that points to all of the new `subclass` is returned

    It is illegal for a tenable buffer to own another tenable buffer. When
    representing the same memory, we should have a single tenable buffer
    and multiple buffer slices.

    Developer Notes
    ---------------
    This function always returns slices thus all buffers in cudf will use
    `BufferSlice` when copy-on-write is enabled. The slices implements
    copy-on-write by trigging deep copies when write access is detected
    and multiple slices points to the same tenable buffer.

    Parameters
    ----------
    data : buffer-like or array-like
        A buffer-like or array-like object that represent C-contiguous memory.
    exposed
        Mark the buffer as permanently exposed.
    subclass
        If not None, a subclass of TenableBuffer to wrap `data`.

    Return
    ------
    BufferSlice
        A buffer slice that points to a TenableBuffer (or `subclass`), which
        in turn wraps `data`.
    """

    if not hasattr(data, "__cuda_array_interface__"):
        if exposed:
            raise ValueError("cannot created exposed host memory")
        return cast(BufferSlice, TenableBuffer._from_host_memory(data)[:])

    owner = get_owner(data, subclass or TenableBuffer)
    if owner is None:
        return cast(
            BufferSlice,
            TenableBuffer._from_device_memory(data, exposed=exposed)[:],
        )

    # At this point, we know that `data` is owned by a tenable buffer
    ptr, size = get_ptr_and_size(data.__cuda_array_interface__)
    if size > 0 and owner._ptr == 0:
        raise ValueError("Cannot create a non-empty slice of a null buffer")
    return BufferSlice(base=owner, offset=ptr - owner._ptr, size=size)


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
        The current exposure status of the buffer. Notice, once the exposure
        status becomes True, it should never change back.
    _slices
        The set of BufferSlice instances that point to this buffer.
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

    @property
    def __cuda_array_interface__(self) -> Mapping:
        self.mark_exposed()
        return super().__cuda_array_interface__

    def __repr__(self) -> str:
        return (
            f"<TenableBuffer exposed={self.exposed} "
            f"size={format_bytes(self._size)} "
            f"ptr={hex(self._ptr)} owner={repr(self._owner)}>"
        )


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
        Name of attributes that are passed through to the base as-is.
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

    def get_ptr(self, *, mode: str) -> int:
        if mode == "write" and cudf.get_option("copy_on_write"):
            self.make_single_owner_inplace()
        return self._base.get_ptr(mode=mode) + self._offset

    def memoryview(
        self, *, offset: int = 0, size: Optional[int] = None
    ) -> memoryview:
        return self._base.memoryview(offset=self._offset + offset, size=size)

    def copy(self, deep: bool = True) -> BufferSlice:
        """Return a copy of Buffer.

        What actually happens when `deep == False` depends on the
        "copy_on_write" option. When copy-on-write is enabled, a shallow copy
        becomes a deep copy if the buffer has been exposed. This is because we
        have no control over knowing if the data is being modified when the
        buffer has been exposed to third-party.

        Parameters
        ----------
        deep : bool, default True
            The semantics when copy-on-write is disabled:
                - If True, returns a deep copy of the underlying Buffer data.
                - If False, returns a shallow copy of the Buffer pointing to
                  the same underlying data.
            The semantics when copy-on-write is enabled:
                - Always a deep copy of the underlying Buffer data.

        Returns
        -------
        BufferSlice
            A slice pointing to either a new or the existing base buffer
            depending on the expose status of the base buffer and the
            copy-on-write option (see above).
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
        """Make sure this slice is the only one pointing to the base.

        This is used by copy-on-write to trigger a deep copy when write
        access is detected.

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
            # If this is not the only slice pointing to `self._base`, we
            # point to a new deep copy of the base.
            t = self.copy(deep=True)
            self._base = t._base
            self._offset = t._offset
            self._size = t._size
            self._owner = t._base
            self._base._slices.add(self)

    def __repr__(self) -> str:
        return (
            f"<BufferSlice size={format_bytes(self._size)} "
            f"offset={format_bytes(self._offset)} of {self._base}>"
        )
