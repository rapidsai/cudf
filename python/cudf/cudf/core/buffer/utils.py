# Copyright (c) 2022, NVIDIA CORPORATION.

from __future__ import annotations

import functools
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
)

from cudf.core.buffer.buffer import (
    Buffer,
    DeviceBufferLike,
    ensure_buffer_like,
)
from cudf.core.buffer.spill_manager import global_manager_get
from cudf.core.buffer.spillable_buffer import SpillableBuffer, SpillLock

if TYPE_CHECKING:
    from cudf._lib.column import Column


def as_device_buffer_like(
    data: Union[int, Any],
    *,
    exposed=True,
    size: int = None,
    owner: object = None,
) -> DeviceBufferLike:
    """
    Factory function to wrap `data` in a DeviceBufferLike object.

    If `data` isn't device-buffer-like already, a new buffer that implements
    DeviceBufferLike and points to the memory of `data` is created. If `data`
    represents host memory, it is copied to a new `rmm.DeviceBuffer` device
    allocation. Otherwise, the data of `data` is **not** copied, instead the
    new buffer keeps a reference to `data` in order to retain the lifetime
    of `data`.

    If `data` is an integer it must represent a device pointer and `size` must
    be specified. If `data` isn't an integer both `size` and `owner` must be
    None.

    Raises ValueError if the data of `data` isn't C-contiguous.

    Parameters
    ----------
    data : int or buffer-like or array-like
        - An integer representing a pointer to device memory, or
        - An object that exposes either device or host memory through
          `__array_interface__`, `__cuda_array_interface__`, or the buffer
          protocol.
        If `data` represents host memory, data will be copied to device.
    exposed : bool, optional
        Whether or not a raw pointer (integer or C pointer) has
        been exposed to the outside world. If this is the case,
        the buffer cannot be spilled.
    size : int, optional
        Size of memory in bytes. Must be specified if `data` is an integer
        otherwise it must be None.
    owner : object, optional
        Python object to which the lifetime of the memory allocation is tied.
        A reference to this object is kept in the returned Buffer. Can only be
        specified when `data` is an integer otherwise the `data` itself will be
        set as the owner.

    Return
    ------
    DeviceBufferLike
        A device-buffer-like instance that represents the device memory
        of `data`.
    """

    if isinstance(data, DeviceBufferLike):
        return data

    data = ensure_buffer_like(data=data, size=size, owner=owner)
    manager = global_manager_get()
    if manager is None:
        return Buffer(data)
    return SpillableBuffer(data=data, exposed=exposed, manager=manager)


def get_columns(obj: Any) -> List[Column]:
    from cudf._lib.column import Column
    from cudf.core.column_accessor import ColumnAccessor
    from cudf.core.frame import Frame
    from cudf.core.indexed_frame import IndexedFrame

    """Return all columns in `obj` (no duplicates)"""
    found: List[Column] = []
    found_ids: Set[int] = set()

    def _get_columns(obj: object) -> None:
        if isinstance(obj, Column):
            if id(obj) not in found_ids:
                found_ids.add(id(obj))
                found.append(obj)
        elif isinstance(obj, IndexedFrame):
            _get_columns(obj._data)
            _get_columns(obj._index)
        elif isinstance(obj, Frame):
            _get_columns(obj._data)
        elif isinstance(obj, ColumnAccessor):
            for o in obj.columns:
                _get_columns(o)
        elif isinstance(obj, (list, tuple)):
            for o in obj:
                _get_columns(o)
        elif isinstance(obj, Mapping):
            for o in obj.values():
                _get_columns(o)

    _get_columns(obj)
    return found


def mark_columns_as_read_only_inplace(obj: Any) -> None:
    """
    Mark all columns found in `obj` as read-only.

    This is an in-place operation, which does nothing if
    spilling is disabled.

    Making columns as read-only makes it possible to unspill the
    underlying buffers partially.
    """

    if global_manager_get() is None:
        return

    for col in get_columns(obj):
        if col.base_children:
            continue  # TODO: support non-fixed-length data types

        if col.base_mask is not None:
            continue  # TODO: support masks

        if col.base_data is None:
            continue
        assert col.data is not None

        if col.data is col.base_data:
            continue  # We can ignore non-views

        if isinstance(col.base_data, SpillableBuffer) and isinstance(
            col.data, SpillableBuffer
        ):
            with col.base_data.lock:
                if not col.base_data.is_spilled:
                    continue  # We can ignore non-spilled columns
                mem = col.data.memoryview()
            col.set_base_data(as_device_buffer_like(mem, exposed=False))
            col._offset = 0


_thread_spill_locks: Dict[int, Tuple[Optional[SpillLock], int]] = {}


def get_spill_lock():
    _id = threading.get_ident()
    spill_lock, _ = _thread_spill_locks.get(_id, (None, 0))
    return spill_lock


def push_thread_spill_lock():
    _id = threading.get_ident()
    spill_lock, count = _thread_spill_locks.get(_id, (None, 0))
    if spill_lock is None:
        spill_lock = SpillLock()
    _thread_spill_locks[_id] = (spill_lock, count + 1)


def pop_thread_spill_lock():
    _id = threading.get_ident()
    spill_lock, count = _thread_spill_locks[_id]
    if count == 1:
        spill_lock = None
    _thread_spill_locks[_id] = (spill_lock, count - 1)


def with_spill_lock(*, read_only_columns=False):
    """Decorator to set spill lock within a function automatically.

    All calls to `SpillManager.get_ptr()` within the decorated function will
    use a spill lock with a lifetime bound to the function execution.

    Parameters
    ----------
    read_only_columns : bool
        Mark all columns found in the arguments to the decorated function
        as read-only. This is an in-place operation, which does nothing if
        spilling is disabled. Making columns as read-only makes it
        possible to unspill the underlying buffers partially.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if global_manager_get() is None:
                # Quick return, if spilling is disabled.
                return func(*args, **kwargs)
            if read_only_columns:
                mark_columns_as_read_only_inplace((args, kwargs))
            push_thread_spill_lock()
            try:
                return func(*args, **kwargs)
            finally:
                pop_thread_spill_lock()

        return wrapper

    return decorator
