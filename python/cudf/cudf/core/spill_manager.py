# Copyright (c) 2022, NVIDIA CORPORATION.

from __future__ import annotations

import gc
import io
import os
import threading
import traceback
import warnings
import weakref
from dataclasses import dataclass
from functools import cached_property
from typing import List, Mapping, MutableMapping, Optional, Set, Tuple

import rmm.mr

from cudf._lib.column import Column
from cudf.core.buffer import DeviceBufferLike, as_device_buffer_like
from cudf.core.column_accessor import ColumnAccessor
from cudf.core.frame import Frame
from cudf.core.indexed_frame import IndexedFrame
from cudf.core.spillable_buffer import SpillableBuffer
from cudf.utils.string import format_bytes


def get_traceback() -> str:
    with io.StringIO() as f:
        traceback.print_stack(file=f)
        f.seek(0)
        return f.read()


@dataclass
class ExposeStatistic:
    traceback: str
    count: int = 1
    total_nbytes: int = 0
    spilled_nbytes: int = 0


class SpillManager:
    _base_buffers: MutableMapping[int, SpillableBuffer]
    _other_buffers: MutableMapping[int, DeviceBufferLike]
    _expose_statistics: Optional[MutableMapping[str, ExposeStatistic]]

    def __init__(
        self,
        *,
        spill_on_demand=False,
        device_memory_limit=None,
        expose_statistics=False,
    ) -> None:
        self._lock = threading.Lock()
        self._base_buffers = weakref.WeakValueDictionary()
        self._other_buffers = weakref.WeakValueDictionary()
        self._id_counter = 0
        self._spill_on_demand = spill_on_demand
        self._device_memory_limit = device_memory_limit
        if self._spill_on_demand:
            self.register_spill_on_demand()
        self._expose_statistics = {} if expose_statistics else None

    def register_spill_on_demand(self):
        # TODO: check if a `FailureCallbackResourceAdaptor` has been
        #       registered already
        def oom(nbytes: int, *, retry_on_error=True) -> bool:
            """Try to handle an out-of-memory error by spilling

            Warning: in order to avoid deadlock, this function should
            not lock already locked buffers.
            """

            # Keep spilling until `nbytes` been spilled
            total_spilled = 0
            while total_spilled < nbytes:
                spilled = self.spill_device_memory()
                if spilled == 0:
                    break  # No more to spill!
                total_spilled += spilled

            if total_spilled > 0:
                return True  # Ask RMM to retry the allocation

            if retry_on_error:
                # Let's collect garbage and try one more time
                gc.collect()
                return oom(nbytes, retry_on_error=False)

            # TODO: write to log instead of stdout
            print(
                f"[WARNING] RMM allocation of {format_bytes(nbytes)} bytes "
                "failed, spill-on-demand couldn't find any device memory to "
                f"spill:\n{repr(self)}\ntraceback:\n{get_traceback()}"
            )
            if self._expose_statistics is None:
                print("Set `CUDF_SPILL_STAT_EXPOSE=on` for expose statistics")
            else:
                print(self.pprint_expose_statistics())

            return False  # Since we didn't find anything to spill, we give up

        current_mr = rmm.mr.get_current_device_resource()
        mr = rmm.mr.FailureCallbackResourceAdaptor(current_mr, oom)
        rmm.mr.set_current_device_resource(mr)

    def add(self, buffer: SpillableBuffer) -> None:
        if buffer.size > 0 and not buffer.exposed:
            with self._lock:
                self._base_buffers[self._id_counter] = buffer
                self._id_counter += 1

    def add_other(self, buffer: DeviceBufferLike) -> None:
        if buffer.size > 0:
            with self._lock:
                self._other_buffers[self._id_counter] = buffer
                self._id_counter += 1

    def base_buffers(
        self, order_by_access_time: bool = False
    ) -> Tuple[SpillableBuffer, ...]:
        with self._lock:
            ret = tuple(self._base_buffers.values())
        if order_by_access_time:
            ret = tuple(sorted(ret, key=lambda b: b.last_accessed))
        return ret

    def other_buffers(self) -> Tuple[DeviceBufferLike, ...]:
        with self._lock:
            ret = tuple(self._other_buffers.values())
        return ret

    def spilled_and_unspilled(self) -> Tuple[int, int]:
        spilled, unspilled = 0, 0
        for buf in self.base_buffers():
            if buf.is_spilled:
                spilled += buf.size
            else:
                unspilled += buf.size
        return spilled, unspilled

    def spill_device_memory(self) -> int:
        """Try to spill device memory

        This function is safe to call doing spill-on-demand
        since it does not lock buffers already locked.

        Return
        ------
        int
            Number of bytes spilled.
        """
        for buf in self.base_buffers(order_by_access_time=True):
            if buf.lock.acquire(blocking=False):
                try:
                    if not buf.is_spilled and buf.spillable:
                        buf.__spill__(target="cpu")
                        return buf.size
                finally:
                    buf.lock.release()
        return 0

    def spill_to_device_limit(self, device_limit: int = None) -> int:
        limit = (
            self._device_memory_limit if device_limit is None else device_limit
        )
        if limit is None:
            return 0
        ret = 0
        while True:
            _, unspilled = self.spilled_and_unspilled()
            if unspilled < limit:
                break
            nbytes = self.spill_device_memory()
            if nbytes == 0:
                break  # No more to spill
            ret += nbytes
        return ret

    def lookup_address_range(
        self, ptr: int, size: int
    ) -> Optional[SpillableBuffer]:
        for buf in self.base_buffers():
            if buf.is_overlapping(ptr, size):
                return buf
        return None

    def log_expose(self, buf: SpillableBuffer) -> None:
        if self._expose_statistics is None:
            return
        tb = get_traceback()
        stat = self._expose_statistics.get(tb, None)
        spilled_nbytes = buf.nbytes if buf.is_spilled else 0
        if stat is None:
            self._expose_statistics[tb] = ExposeStatistic(
                traceback=tb,
                total_nbytes=buf.nbytes,
                spilled_nbytes=spilled_nbytes,
            )
        else:
            stat.count += 1
            stat.total_nbytes += buf.nbytes
            stat.spilled_nbytes += spilled_nbytes

    def get_expose_statistics(self) -> List[ExposeStatistic]:
        if self._expose_statistics is None:
            return []
        return sorted(self._expose_statistics.values(), key=lambda x: -x.count)

    def pprint_expose_statistics(self) -> str:
        ret = "Expose Statistics:\n"
        for s in self.get_expose_statistics():
            ret += (
                f" Count: {s.count}, total: {format_bytes(s.total_nbytes)}, "
            )
            ret += f"spilled: {format_bytes(s.spilled_nbytes)}\n"
            ret += s.traceback
            ret += "\n"
        return ret

    def __repr__(self) -> str:
        spilled, unspilled = self.spilled_and_unspilled()

        unspillable = 0
        for buf in self.base_buffers():
            if not (buf.is_spilled or buf.spillable):
                unspillable += buf.size
        unspillable_ratio = unspillable / unspilled if unspilled else 0

        others = sum(b.size for b in self.other_buffers())
        return (
            f"<SpillManager spill_on_demand={self._spill_on_demand} "
            f"device_memory_limit={self._device_memory_limit} | "
            f"{format_bytes(spilled)} spilled | "
            f"{format_bytes(unspilled)} ({unspillable_ratio:.0%}) "
            f"unspilled (unspillable) | {format_bytes(others)} others>"
        )


# TODO: do we have a common "get-value-from-env" in cuDF?
def _env_get_int(name, default):
    try:
        return int(os.getenv(name, default))
    except (ValueError, TypeError):
        return default


def _env_get_bool(name, default):
    env = os.getenv(name)
    if env is None:
        return default
    as_a_int = _env_get_int(name, None)
    env = env.lower().strip()
    if env == "true" or env == "on" or as_a_int:
        return True
    if env == "false" or env == "off" or as_a_int == 0:
        return False
    return default


class GlobalSpillManager:
    def __init__(self):
        self._manager: Optional[SpillManager] = None

    def get_manager_from_env(self) -> Optional[SpillManager]:
        if not _env_get_bool("CUDF_SPILL", False):
            return None
        return SpillManager(
            spill_on_demand=_env_get_bool("CUDF_SPILL_ON_DEMAND", True),
            device_memory_limit=_env_get_int("CUDF_SPILL_DEVICE_LIMIT", None),
            expose_statistics=_env_get_bool("CUDF_SPILL_STAT_EXPOSE", False),
        )

    def clear(self) -> None:
        if self._manager is not None:
            gc.collect()
            base_buffers = self._manager.base_buffers()
            if len(base_buffers) > 0:
                warnings.warn(f"overwriting non-empty manager: {base_buffers}")
        self._manager = None
        self.__dict__.pop("enabled", None)  # Clear cache

    def reset(self, manager: SpillManager) -> SpillManager:
        self.clear()
        self._manager = manager
        return manager

    @cached_property
    def enabled(self) -> bool:
        if self._manager is None:
            self._manager = self.get_manager_from_env()
        return self._manager is not None

    def get(self) -> SpillManager:
        if self.enabled:
            assert isinstance(self._manager, SpillManager)
            return self._manager
        raise ValueError("No global SpillManager")


global_manager = GlobalSpillManager()


def get_columns(obj: object) -> List[Column]:
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
