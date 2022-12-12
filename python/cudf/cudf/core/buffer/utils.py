# Copyright (c) 2022, NVIDIA CORPORATION.

from __future__ import annotations

import functools
import sys
import threading
import weakref
from contextlib import ContextDecorator
from typing import Any, Dict, Optional, Tuple, TypeVar, Union

import cudf
from cudf.core.buffer.buffer import Buffer, cuda_array_interface_wrapper
from cudf.core.buffer.spill_manager import get_global_manager
from cudf.core.buffer.spillable_buffer import SpillableBuffer, SpillLock

T = TypeVar("T")


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


def _clear_property_cache(
    instance_ref: weakref.ReferenceType[T], nbytes: int, attrname: str
) -> Optional[int]:
    """Spill handler that clears the `cached_property` of an instance

    The signature of this function is compatible with SpillManager's
    register_spill_handler.

    To avoid keeping instance alive, we take a weak reference of the instance.

    Parameters
    ----------
    instance_ref
        Weakref of the instance
    nbytes : int
        Size of the cached data
    attrname : str
        Name of the cached attribute

    Return
    ------
    int
        Number of bytes cleared
    """

    instance = instance_ref()
    if instance is None:
        return 0

    cached = instance.__dict__.get(attrname, None)
    if cached is None:
        return None  # The cached has been cleared

    # If `cached` is known outside of the cache, we cannot free any
    # memory by clearing the cache. We have three inside references:
    # `instance.__dict__`, `cached`, and `sys.getrefcount`.
    if sys.getrefcount(cached) > 3:
        return None

    instance.__dict__.pop(attrname, None)  # Clear cache atomically
    return nbytes


class cached_property(functools.cached_property):
    """A version of `cached_property` that delete instead of spill the cache

    When spilling is disabled (the default case), this decorator is identical
    to `functools.cached_property`.

    When spilling is enabled, this property register a spill handler for
    the cached data that deletes the data rather than spilling it. For now,
    only cached Columns are handled this way.
    See `SpillManager.register_spill_handler`.
    """

    def __get__(self, instance: T, owner=None):
        cache_hit = self.attrname in instance.__dict__
        ret = super().__get__(instance, owner)
        if cache_hit or not isinstance(ret, cudf.core.column.ColumnBase):
            return ret

        manager = get_global_manager()
        if manager is None:
            return ret

        buf = ret.base_data
        if buf is None or buf.nbytes == 0:
            return ret
        assert isinstance(buf, SpillableBuffer)

        manager.register_spill_handler(
            buf,
            _clear_property_cache,
            weakref.ref(instance),
            nbytes=buf.nbytes,
            attrname=self.attrname,
        )
        return ret
