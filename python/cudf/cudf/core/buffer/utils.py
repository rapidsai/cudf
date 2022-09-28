# Copyright (c) 2022, NVIDIA CORPORATION.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Mapping, Set

from cudf.core.buffer.buffer import Buffer, DeviceBufferLike
from cudf.core.buffer.spill_manager import global_manager
from cudf.core.buffer.spillable_buffer import SpillableBuffer

if TYPE_CHECKING:
    from cudf._lib.column import Column


def as_device_buffer_like(obj: Any, exposed=True) -> DeviceBufferLike:
    """
    Factory function to wrap `obj` in a DeviceBufferLike object.

    If `obj` isn't device-buffer-like already, a new buffer that implements
    DeviceBufferLike and points to the memory of `obj` is created. If `obj`
    represents host memory, it is copied to a new `rmm.DeviceBuffer` device
    allocation. Otherwise, the data of `obj` is **not** copied, instead the
    new buffer keeps a reference to `obj` in order to retain the lifetime
    of `obj`.

    Raises ValueError if the data of `obj` isn't C-contiguous.

    Parameters
    ----------
    obj : buffer-like or array-like
        An object that exposes either device or host memory through
        `__array_interface__`, `__cuda_array_interface__`, or the
        buffer protocol. If `obj` represents host memory, data will
        be copied.
    exposed : bool, optional
        Whether or not a raw pointer (integer or C pointer) has
        been exposed to the outside world. If this is the case,
        the buffer cannot be spilled.

    Return
    ------
    DeviceBufferLike
        A device-buffer-like instance that represents the device memory
        of `obj`.
    """

    if isinstance(obj, DeviceBufferLike):
        return obj

    if global_manager.enabled:
        return SpillableBuffer(
            data=obj, exposed=exposed, manager=global_manager.get()
        )
    else:
        return Buffer(obj)


def get_columns(obj: object) -> List[Column]:
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


def mark_columns_as_read_only_inplace(obj: object) -> None:
    """
    Mark all columns found in `obj` as read-only.

    This is an in-place operation, which does nothing if
    spilling is disabled.

    Making columns as ready-only, makes it possible to unspill the
    underlying buffers partially.
    """

    if not global_manager.enabled:
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
