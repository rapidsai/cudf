# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from __future__ import annotations

import weakref
from typing import (
    Any,
    Container,
    Literal,
    Mapping,
    Optional,
    Type,
    TypeVar,
    cast,
)

from typing_extensions import Self

import cudf
from cudf.core.buffer.buffer import Buffer, get_ptr_and_size
from cudf.utils.string import format_bytes

T = TypeVar("T", bound="ExposureTrackedBuffer")


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


def as_exposure_tracked_buffer(
    data, exposed: bool, subclass: Optional[Type[T]] = None
) -> BufferSlice:
    """Factory function to wrap `data` in a slice of an exposure tracked buffer

    If `subclass` is None, a new ExposureTrackedBuffer that points to the
    memory of `data` is created and a BufferSlice that points to all of the
    new ExposureTrackedBuffer is returned.

    If `subclass` is not None, a new `subclass` is created instead. Still,
    a BufferSlice that points to all of the new `subclass` is returned

    It is illegal for an exposure tracked buffer to own another exposure
    tracked buffer. When representing the same memory, we should have a single
    exposure tracked buffer and multiple buffer slices.

    Developer Notes
    ---------------
    This function always returns slices thus all buffers in cudf will use
    `BufferSlice` when copy-on-write is enabled. The slices implement
    copy-on-write by trigging deep copies when write access is detected
    and multiple slices points to the same exposure tracked buffer.

    Parameters
    ----------
    data : buffer-like or array-like
        A buffer-like or array-like object that represents C-contiguous memory.
    exposed
        Mark the buffer as permanently exposed.
    subclass
        If not None, a subclass of ExposureTrackedBuffer to wrap `data`.

    Return
    ------
    BufferSlice
        A buffer slice that points to a ExposureTrackedBuffer (or `subclass`),
        which in turn wraps `data`.
    """

    if not hasattr(data, "__cuda_array_interface__"):
        if exposed:
            raise ValueError("cannot created exposed host memory")
        return cast(
            BufferSlice, ExposureTrackedBuffer._from_host_memory(data)[:]
        )

    owner = get_owner(data, subclass or ExposureTrackedBuffer)
    if owner is None:
        return cast(
            BufferSlice,
            ExposureTrackedBuffer._from_device_memory(data, exposed=exposed)[
                :
            ],
        )

    # At this point, we know that `data` is owned by a exposure tracked buffer
    ptr, size = get_ptr_and_size(data.__cuda_array_interface__)
    if size > 0 and owner._ptr == 0:
        raise ValueError("Cannot create a non-empty slice of a null buffer")
    return BufferSlice(base=owner, offset=ptr - owner._ptr, size=size)


class ExposureTrackedBuffer(Buffer):
    """A Buffer that tracks its "expose" status.

    In order to implement copy-on-write and spillable buffers, we need the
    ability to detect external access to the underlying memory. We say that
    the buffer has been exposed if the device pointer (integer or void*) has
    been accessed outside of ExposureTrackedBuffer. In this case, we have no
    control over knowing if the data is being modified by a third-party.

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
    def _from_device_memory(cls, data: Any, *, exposed: bool = False) -> Self:
        """Create an exposure tracked buffer from device memory.

        No data is being copied.

        Parameters
        ----------
        data : device-buffer-like
            An object implementing the CUDA Array Interface.
        exposed : bool, optional
            Mark the buffer as permanently exposed.

        Returns
        -------
        ExposureTrackedBuffer
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
            f"<ExposureTrackedBuffer exposed={self.exposed} "
            f"size={format_bytes(self._size)} "
            f"ptr={hex(self._ptr)} owner={repr(self._owner)}>"
        )


class BufferSlice(ExposureTrackedBuffer):
    """A slice (aka. a view) of a exposure tracked buffer.

    Parameters
    ----------
    base
        The exposure tracked buffer this slice refers to.
    offset
        The offset relative to the start memory of base (in bytes).
    size
        The size of the slice (in bytes)
    passthrough_attributes
        Name of attributes that are passed through to the base as-is.
    """

    def __init__(
        self,
        base: ExposureTrackedBuffer,
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

    def get_ptr(self, *, mode: Literal["read", "write"]) -> int:
        if mode == "write" and cudf.get_option("copy_on_write"):
            self.make_single_owner_inplace()
        return self._base.get_ptr(mode=mode) + self._offset

    def memoryview(
        self, *, offset: int = 0, size: Optional[int] = None
    ) -> memoryview:
        return self._base.memoryview(offset=self._offset + offset, size=size)

    def copy(self, deep: bool = True) -> Self:
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
                - If deep=True, returns a deep copy of the underlying data.
                - If deep=False, returns a shallow copy of the Buffer pointing
                  to the same underlying data.
            The semantics when copy-on-write is enabled:
                - From the users perspective, always a deep copy of the
                  underlying data. However, the data isn't actually copied
                  until someone writers to the returned buffer.

        Returns
        -------
        BufferSlice
            A slice pointing to either a new or the existing base buffer
            depending on the expose status of the base buffer and the
            copy-on-write option (see above).
        """
        if cudf.get_option("copy_on_write"):
            base_copy = self._base.copy(deep=deep or self.exposed)
        else:
            base_copy = self._base.copy(deep=deep)
        return cast(Self, base_copy[self._offset : self._offset + self._size])

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
