# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from cudf.core.buffer.buffer import (
    Buffer,
    BufferOwner,
    get_ptr_and_size,
)
from cudf.core.buffer.spill_manager import get_global_manager
from cudf.core.buffer.spillable_buffer import (
    SpillableBuffer,
    SpillableBufferOwner,
)


def get_buffer_owner(data: Any) -> BufferOwner | None:
    """Get the owner of `data`, if one exists

    Search through the stack of data owners in order to find an
    owner BufferOwner (incl. subclasses).

    Parameters
    ----------
    data
        The data object to search for a BufferOwner instance

    Return
    ------
    BufferOwner or None
        The owner of `data` if found otherwise None.
    """

    if isinstance(data, BufferOwner):
        return data
    if hasattr(data, "owner"):
        return get_buffer_owner(data.owner)
    return None


def as_buffer(
    data: Any,
) -> Buffer:
    """Factory function to wrap `data` in a Buffer object.

    If `data` isn't a buffer already, a new buffer that points to the memory of `data`
    is created. `data` must either be convertible to a numpy array (for host memory) or
    satisfy the CUDA Array Interface for device memory. If `data` represents host
    memory, it is copied to a new `rmm.DeviceBuffer` device allocation. Otherwise, the
    memory of `data` is **not** copied, instead the new buffer keeps a reference to
    `data` in order to retain its lifetime.

    Raises ValueError if `data` isn't C-contiguous.

    If spilling is enabled, a SpillableBuffer that refers to a
    SpillableBufferOwner is returned. If `data` is owned by a spillable buffer,
    it must either be "exposed" or accessed within a buffer/column access context.
    This is to guarantee that the memory of `data` isn't spilled before this
    function gets to calculate the offset of the new SpillableBuffer.


    Parameters
    ----------
    data : buffer-like or array-like
        A buffer-like or array-like object.

    Return
    ------
    Buffer
        A buffer instance that represents the device memory of `data`.
    """

    if isinstance(data, Buffer):
        return data

    # Find the buffer types to return based on the current config
    owner_class: type[BufferOwner]
    buffer_class: type[Buffer]
    if get_global_manager() is not None:
        owner_class = SpillableBufferOwner
        buffer_class = SpillableBuffer
    else:
        owner_class = BufferOwner
        buffer_class = Buffer

    # Handle host memory,
    if isinstance(data, memoryview):
        return buffer_class(owner=owner_class.from_host_memory(data))
    elif not hasattr(data, "__cuda_array_interface__"):
        raise ValueError(
            "data must be a Buffer, memoryview, or implement __cuda_array_interface__"
        )

    # Check if `data` is owned by a known class
    owner = get_buffer_owner(data)
    if owner is None:  # `data` is new device memory
        return buffer_class(owner=owner_class.from_device_memory(data))

    # At this point, we know that `data` is owned by a known class, which
    # should be the same class as specified by the current config (see above)
    assert owner.__class__ is owner_class
    if (
        isinstance(owner, SpillableBufferOwner)
        and not owner.exposed
        and not owner._spill_locks
    ):
        raise ValueError(
            "An owning spillable buffer must "
            "either be exposed or spill locked."
        )
    ptr, size = get_ptr_and_size(data.__cuda_array_interface__)
    base_ptr = owner.ptr
    if size > 0 and base_ptr == 0:
        raise ValueError("Cannot create a non-empty slice of a null buffer")
    return buffer_class(owner=owner, offset=ptr - base_ptr, size=size)
