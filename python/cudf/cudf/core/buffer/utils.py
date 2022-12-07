# Copyright (c) 2022, NVIDIA CORPORATION.

from __future__ import annotations

import threading
from contextlib import ContextDecorator
from typing import Any, Dict, Optional, Tuple, Union

from cudf.core.buffer.buffer import Buffer, cuda_array_interface_wrapper
from cudf.core.buffer.spill_manager import get_global_manager
from cudf.core.buffer.spillable_buffer import SpillableBuffer, SpillLock


def as_buffer(
    data: Union[int, Any],
    *,
    size: int = None,
    owner: object = None,
    exposed: bool = False,
) -> Buffer:
    """Factory function to wrap `data` in a Buffer object.

    If `data` isn't a buffer already, a new buffer that points to the memory of
    `data` is created. If `data` represents host memory, it is copied to a new
    `rmm.DeviceBuffer` device allocation. Otherwise, the memory of `data` is
    **not** copied, instead the new buffer keeps a reference to `data` in order
    to retain its lifetime.

    If `data` is an integer, it is assumed to point to device memory.

    Raises ValueError if data isn't C-contiguous.

    Parameters
    ----------
    data : int or buffer-like or array-like
        An integer representing a pointer to device memory or a buffer-like
        or array-like object. When not an integer, `size` and `owner` must
        be None.
    size : int, optional
        Size of device memory in bytes. Must be specified if `data` is an
        integer.
    owner : object, optional
        Python object to which the lifetime of the memory allocation is tied.
        A reference to this object is kept in the returned Buffer.
    exposed : bool, optional
        Mark the buffer as permanently exposed (unspillable). This is ignored
        unless spilling is enabled and the data represents device memory, see
        SpillableBuffer.

    Return
    ------
    Buffer
        A buffer instance that represents the device memory of `data`.
    """

    if isinstance(data, Buffer):
        return data

    # We handle the integer argument in the factory function by wrapping
    # the pointer in a `__cuda_array_interface__` exposing object so that
    # the Buffer (and its sub-classes) do not have to.
    if isinstance(data, int):
        if size is None:
            raise ValueError(
                "size must be specified when `data` is an integer"
            )
        data = cuda_array_interface_wrapper(ptr=data, size=size, owner=owner)
    elif size is not None or owner is not None:
        raise ValueError(
            "`size` and `owner` must be None when "
            "`data` is a buffer-like or array-like object"
        )

    if get_global_manager() is not None:
        if hasattr(data, "__cuda_array_interface__"):
            return SpillableBuffer._from_device_memory(data, exposed=exposed)
        if exposed:
            raise ValueError("cannot created exposed host memory")
        return SpillableBuffer._from_host_memory(data)

    if hasattr(data, "__cuda_array_interface__"):
        return Buffer._from_device_memory(data)
    return Buffer._from_host_memory(data)


_thread_spill_locks: Dict[int, Tuple[Optional[SpillLock], int]] = {}


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

    def __enter__(self) -> Optional[SpillLock]:
        _push_thread_spill_lock()
        return get_spill_lock()

    def __exit__(self, *exc):
        _pop_thread_spill_lock()


def get_spill_lock() -> Union[SpillLock, None]:
    """Return a spill lock within the context of `acquire_spill_lock` or None

    Returns None, if spilling is disabled.
    """

    if get_global_manager() is None:
        return None
    _id = threading.get_ident()
    spill_lock, _ = _thread_spill_locks.get(_id, (None, 0))
    return spill_lock
