# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from contextlib import ContextDecorator
from typing import Any

from cudf.core.buffer.buffer import (
    Buffer,
    BufferOwner,
    get_ptr_and_size,
)
from cudf.core.buffer.exposure_tracked_buffer import ExposureTrackedBuffer
from cudf.core.buffer.spill_manager import get_global_manager
from cudf.core.buffer.spillable_buffer import (
    SpillableBuffer,
    SpillableBufferOwner,
    SpillLock,
)
from cudf.options import get_option


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
    *,
    exposed: bool = False,
) -> Buffer:
    """Factory function to wrap `data` in a Buffer object.

    If `data` isn't a buffer already, a new buffer that points to the memory of `data`
    is created. `data` must either be convertible to a numpy array (for host memory) or
    satisfy the CUDA Array Interface for device memory. If `data` represents host
    memory, it is copied to a new `rmm.DeviceBuffer` device allocation. Otherwise, the
    memory of `data` is **not** copied, instead the new buffer keeps a reference to
    `data` in order to retain its lifetime.

    Raises ValueError if `data` isn't C-contiguous.

    If copy-on-write is enabled, an ExposureTrackedBuffer is returned.

    If spilling is enabled, a SpillableBuffer that refers to a
    SpillableBufferOwner is returned. If `data` is owned by a spillable buffer,
    it must either be "exposed" or spill locked (called within an
    acquire_spill_lock context). This is to guarantee that the memory of `data`
    isn't spilled before this function gets to calculate the offset of the new
    SpillableBuffer.


    Parameters
    ----------
    data : buffer-like or array-like
        A buffer-like or array-like object.
    exposed : bool, optional
        Mark the buffer as permanently exposed. This is used by
        ExposureTrackedBuffer to determine when a deep copy is required and
        by SpillableBuffer to mark the buffer unspillable.

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
    elif get_option("copy_on_write"):
        owner_class = BufferOwner
        buffer_class = ExposureTrackedBuffer
    else:
        owner_class = BufferOwner
        buffer_class = Buffer

    # Handle host memory,
    if not hasattr(data, "__cuda_array_interface__"):
        if exposed:
            raise ValueError("cannot created exposed host memory")
        return buffer_class(owner=owner_class.from_host_memory(data))

    # Check if `data` is owned by a known class
    owner = get_buffer_owner(data)
    if owner is None:  # `data` is new device memory
        return buffer_class(
            owner=owner_class.from_device_memory(data, exposed=exposed)
        )

    # At this point, we know that `data` is owned by a known class, which
    # should be the same class as specified by the current config (see above)
    assert owner.__class__ is owner_class
    if (
        isinstance(owner, SpillableBufferOwner)
        and not owner.exposed
        and get_spill_lock() is None
    ):
        raise ValueError(
            "An owning spillable buffer must "
            "either be exposed or spill locked."
        )
    ptr, size = get_ptr_and_size(data.__cuda_array_interface__)
    base_ptr = owner.get_ptr(mode="read")
    if size > 0 and base_ptr == 0:
        raise ValueError("Cannot create a non-empty slice of a null buffer")
    return buffer_class(owner=owner, offset=ptr - base_ptr, size=size)


_thread_spill_locks: dict[int, tuple[SpillLock | None, int]] = {}


def _push_thread_spill_lock() -> None:
    _id = threading.get_ident()
    spill_lock, count = _thread_spill_locks.get(_id, (None, 0))
    if spill_lock is None:
        spill_lock = SpillLock()
    _thread_spill_locks[_id] = (spill_lock, count + 1)


def _pop_thread_spill_lock() -> None:
    _id = threading.get_ident()
    spill_lock, count = _thread_spill_locks[_id]
    if count == 1:
        spill_lock = None
    _thread_spill_locks[_id] = (spill_lock, count - 1)


class acquire_spill_lock(ContextDecorator):
    """Decorator and context to set spill lock automatically.

    All calls to `get_spill_lock()` within the decorated function or context
    will return a spill lock with a lifetime bound to the function or context.

    Developer Notes
    ---------------
    We use the global variable `_thread_spill_locks` to track the global spill
    lock state. To support concurrency, each thread tracks its own state by
    pushing and popping from `_thread_spill_locks` using its thread ID.
    """

    def __enter__(self) -> SpillLock | None:
        _push_thread_spill_lock()
        return get_spill_lock()

    def __exit__(self, *exc):
        _pop_thread_spill_lock()


def get_spill_lock() -> SpillLock | None:
    """Return a spill lock within the context of `acquire_spill_lock` or None

    Returns None, if spilling is disabled.
    """

    if get_global_manager() is None:
        return None
    _id = threading.get_ident()
    spill_lock, _ = _thread_spill_locks.get(_id, (None, 0))
    return spill_lock
