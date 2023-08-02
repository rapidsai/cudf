# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from __future__ import annotations

import weakref
from typing import Any, Literal, Mapping, Optional, Type

from typing_extensions import Self

import cudf
from cudf.core.buffer.buffer import (
    Buffer,
    BufferOwner,
    get_owner,
    get_ptr_and_size,
)


def as_exposure_tracked_buffer(
    data, exposed: bool, subclass: Optional[Type[ExposureTrackedBuffer]] = None
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
        tracked_buf = ExposureTrackedBuffer._from_host_memory(data)
        return BufferSlice(owner=tracked_buf, offset=0, size=tracked_buf.size)

    owner = get_owner(data, subclass or ExposureTrackedBuffer)
    if owner is not None:
        # `data` is owned by an exposure tracked buffer
        ptr, size = get_ptr_and_size(data.__cuda_array_interface__)
        base_ptr = owner.get_ptr(mode="read")
        if size > 0 and base_ptr == 0:
            raise ValueError(
                "Cannot create a non-empty slice of a null buffer"
            )
        return BufferSlice(owner=owner, offset=ptr - base_ptr, size=size)

    # `data` is new device memory
    owner = ExposureTrackedBuffer._from_device_memory(data, exposed=exposed)
    return BufferSlice(owner=owner, offset=0, size=owner.size)


class ExposureTrackedBuffer(BufferOwner):
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
        ret = super()._from_device_memory(data)
        ret._exposed = exposed
        ret._slices = weakref.WeakSet()
        return ret

    @property
    def __cuda_array_interface__(self) -> Mapping:
        self.mark_exposed()
        return super().__cuda_array_interface__


class BufferSlice(Buffer):
    """A slice (aka. a view) of a exposure tracked buffer.

    Parameters
    ----------
    owner
        The exposure tracked buffer this slice refers to.
    offset
        The offset relative to the start memory of owner (in bytes).
    size
        The size of the slice (in bytes)
    """

    _owner: ExposureTrackedBuffer

    def __init__(
        self,
        owner: ExposureTrackedBuffer,
        offset: int,
        size: int,
    ) -> None:
        super().__init__(owner=owner, offset=offset, size=size)
        self._owner._slices.add(self)

    @property
    def exposed(self) -> bool:
        return self._owner.exposed

    def get_ptr(self, *, mode: Literal["read", "write"]) -> int:
        if mode == "write" and cudf.get_option("copy_on_write"):
            self.make_single_owner_inplace()
        return super().get_ptr(mode=mode)

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
            A slice pointing to either a new or the existing owner
            depending on the expose status of the owner and the
            copy-on-write option (see above).
        """
        if cudf.get_option("copy_on_write"):
            return super().copy(deep=deep or self.exposed)
        return super().copy(deep=deep)

    @property
    def __cuda_array_interface__(self) -> Mapping:
        if cudf.get_option("copy_on_write"):
            self.make_single_owner_inplace()
        return super().__cuda_array_interface__

    def make_single_owner_inplace(self) -> None:
        """Make sure this slice is the only one pointing to the owner.

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

        if len(self._owner._slices) > 1:
            # If this is not the only slice pointing to `self._owner`, we
            # point to a new deep copy of the owner.
            t = self.copy(deep=True)
            self._owner = t._owner
            self._offset = t._offset
            self._size = t._size
            self._owner._slices.add(self)
