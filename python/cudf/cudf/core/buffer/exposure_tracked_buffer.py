# Copyright (c) 2020-2023, NVIDIA CORPORATION.

from __future__ import annotations

import weakref
from typing import Any, Literal, Mapping, Optional

from typing_extensions import Self

import cudf
from cudf.core.buffer.buffer import Buffer, BufferOwner


class ExposureTrackedBufferOwner(BufferOwner):
    """A Buffer that tracks its "expose" status.

    In order to implement copy-on-write and spillable buffers, we need the
    ability to detect external access to the underlying memory. We say that
    the buffer has been exposed if the device pointer (integer or void*) has
    been accessed outside of ExposureTrackedBufferOwner. In this case, we have
    no control over knowing if the data is being modified by a third-party.

    Attributes
    ----------
    _exposed
        The current exposure status of the buffer. Notice, once the exposure
        status becomes True, it should never change back.
    _slices
        The set of ExposureTrackedBuffer instances that point to this buffer.
    """

    _exposed: bool
    _slices: weakref.WeakSet[ExposureTrackedBuffer]

    @property
    def exposed(self) -> bool:
        return self._exposed

    def mark_exposed(self) -> None:
        """Mark the buffer as "exposed" permanently"""
        self._exposed = True

    @classmethod
    def _from_device_memory(cls, data: Any, exposed: bool) -> Self:
        ret = super()._from_device_memory(data, exposed=exposed)
        ret._exposed = exposed
        ret._slices = weakref.WeakSet()
        return ret

    @property
    def __cuda_array_interface__(self) -> Mapping:
        self.mark_exposed()
        return super().__cuda_array_interface__


class ExposureTrackedBuffer(Buffer):
    """An exposure tracked buffer.

    Parameters
    ----------
    owner
        The owning exposure tracked buffer this refers to.
    offset
        The offset relative to the start memory of owner (in bytes).
    size
        The size of the slice (in bytes)
    """

    _owner: ExposureTrackedBufferOwner

    def __init__(
        self,
        owner: ExposureTrackedBufferOwner,
        offset: int = 0,
        size: Optional[int] = None,
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
        ExposureTrackedBuffer
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
