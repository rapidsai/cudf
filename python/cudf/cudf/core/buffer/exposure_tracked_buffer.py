# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from typing_extensions import Self

import cudf
from cudf.core.buffer.buffer import Buffer, BufferOwner

if TYPE_CHECKING:
    from collections.abc import Mapping


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

    def __init__(
        self,
        owner: BufferOwner,
        offset: int = 0,
        size: int | None = None,
    ) -> None:
        super().__init__(owner=owner, offset=offset, size=size)
        self.owner._slices.add(self)

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
            return super().copy(deep=deep or self.owner.exposed)
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

        if len(self.owner._slices) > 1:
            # If this is not the only slice pointing to `self.owner`, we
            # point to a new copy of our slice of `self.owner`.
            t = self.copy(deep=True)
            self._owner = t.owner
            self._offset = t._offset
            self._size = t._size
            self._owner._slices.add(self)
