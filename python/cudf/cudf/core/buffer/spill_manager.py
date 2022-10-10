# Copyright (c) 2022, NVIDIA CORPORATION.

from __future__ import annotations

import gc
import io
import os
import threading
import traceback
import warnings
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, MutableMapping, Optional, Tuple

import rmm.mr

from cudf.core.buffer.spillable_buffer import SpillableBuffer
from cudf.utils.string import format_bytes


def get_traceback() -> str:
    with io.StringIO() as f:
        traceback.print_stack(file=f)
        f.seek(0)
        return f.read()


def get_rmm_memory_resource_stack(
    mr: rmm.mr.DeviceMemoryResource,
) -> List[rmm.mr.DeviceMemoryResource]:
    if hasattr(mr, "upstream_mr"):
        return [mr] + get_rmm_memory_resource_stack(mr.upstream_mr)
    return [mr]


@dataclass
class ExposeStatistic:
    traceback: str
    count: int = 1
    total_nbytes: int = 0
    spilled_nbytes: int = 0


class Statistics:
    spills_totals: Dict[Tuple[str, str], Tuple[int, float]]
    expose: Dict[str, ExposeStatistic]

    def __init__(self, level) -> None:
        self.lock = threading.Lock()
        self.level = level
        self.spills_totals = defaultdict(lambda: (0, 0))
        self.expose = {}

    def __str__(self) -> str:
        ret = f"Spill Manager Statistics (level={self.level}):\n"
        if self.level == 0:
            return ret[:-1] + " N/A"

        # Print spilling stats
        ret += " Spilling (level >= 1):"
        if len(self.spills_totals) == 0:
            ret += " None"
        ret += "\n"
        for (src, dst), (nbytes, time) in self.spills_totals.items():
            ret += f"  {src} => {dst}: {format_bytes(nbytes)} in {time}s\n"

        # Print expose stats
        ret += " Expose (level >= 2):"
        if self.level < 2:
            return ret + " disabled"
        if len(self.expose) == 0:
            ret += " None"
        ret += "\n"
        for s in sorted(self.expose.values(), key=lambda x: -x.count):
            ret += (
                f" Count: {s.count}, total: {format_bytes(s.total_nbytes)}, "
            )
            ret += f"spilled: {format_bytes(s.spilled_nbytes)}\n"
            ret += s.traceback
            ret += "\n"
        return ret[:-1]

    def log_spill(self, src: str, dst: str, nbytes: int, time: float) -> None:
        if self.level < 1:
            return
        with self.lock:
            total_nbytes, total_time = self.spills_totals[(src, dst)]
            self.spills_totals[(src, dst)] = (
                total_nbytes + nbytes,
                total_time + time,
            )

    def log_expose(self, buf: SpillableBuffer) -> None:
        if self.level < 2:
            return

        with self.lock:
            tb = get_traceback()
            stat = self.expose.get(tb, None)
            spilled_nbytes = buf.nbytes if buf.is_spilled else 0
            if stat is None:
                self.expose[tb] = ExposeStatistic(
                    traceback=tb,
                    total_nbytes=buf.nbytes,
                    spilled_nbytes=spilled_nbytes,
                )
            else:
                stat.count += 1
                stat.total_nbytes += buf.nbytes
                stat.spilled_nbytes += spilled_nbytes


class SpillManager:
    """Manager of spillable buffers.

    This class implements tracking of all known spillable buffers, on-demand
    spilling of said buffers, and (optional) maintains a memory usage limit.

    When `spill_on_demand=True`, the manager registers an RMM out-of-memory
    error handler, which will spill spillable buffers in order to free up
    memory.

    When `device_memory_limit=True`, the manager will try keep the device
    memory usage below the specified limit by spilling of spillable buffers
    continuously, which will introduce a modest overhead.

    Parameters
    ----------
    spill_on_demand : bool
        Enable spill on demand. The global manager sets this to the value of
        `CUDF_SPILL_ON_DEMAND` or False.
    device_memory_limit: int, optional
        If not None, this is the device memory limit in bytes that triggers
        device to host spilling. The global manager sets this to the value
        of `CUDF_SPILL_DEVICE_LIMIT` or None.
    """

    _base_buffers: MutableMapping[int, SpillableBuffer]
    statistics: Statistics

    def __init__(
        self,
        *,
        spill_on_demand: bool = False,
        device_memory_limit: int = None,
        statistic_level: int = 0,
    ) -> None:
        self._lock = threading.Lock()
        self._base_buffers = weakref.WeakValueDictionary()
        self._id_counter = 0
        self._spill_on_demand = spill_on_demand
        self._device_memory_limit = device_memory_limit
        self.statistics = Statistics(statistic_level)

        if self._spill_on_demand:
            # Set the RMM out-of-memory handle if not already set
            mr = rmm.mr.get_current_device_resource()
            if all(
                not isinstance(m, rmm.mr.FailureCallbackResourceAdaptor)
                for m in get_rmm_memory_resource_stack(mr)
            ):
                rmm.mr.set_current_device_resource(
                    rmm.mr.FailureCallbackResourceAdaptor(
                        mr, self._out_of_memory_handle
                    )
                )

    def _out_of_memory_handle(self, nbytes: int, *, retry_once=True) -> bool:
        """Try to handle an out-of-memory error by spilling

        This can by used as the callback function to RMM's
        `FailureCallbackResourceAdaptor`

        Parameters
        ----------
        nbytes : int
            Number of bytes to try to spill.
        retry_once : bool, optional
            If True, call `gc.collect()` and retry once.

        Return
        ------
        bool
            True if any buffers were freed otherwise False.

        Warning
        -------
        In order to avoid deadlock, this function should not lock
        already locked buffers.
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

        if retry_once:
            # Let's collect garbage and try one more time
            gc.collect()
            return self._out_of_memory_handle(nbytes, retry_once=False)

        # TODO: write to log instead of stdout
        print(
            f"[WARNING] RMM allocation of {format_bytes(nbytes)} bytes "
            "failed, spill-on-demand couldn't find any device memory to "
            f"spill:\n{repr(self)}\ntraceback:\n{get_traceback()}\n"
            f"{self.statistics}"
        )
        return False  # Since we didn't find anything to spill, we give up

    def add(self, buffer: SpillableBuffer) -> None:
        if buffer.size > 0 and not buffer.exposed:
            with self._lock:
                self._base_buffers[self._id_counter] = buffer
                self._id_counter += 1
        self.spill_to_device_limit()

    def base_buffers(
        self, order_by_access_time: bool = False
    ) -> Tuple[SpillableBuffer, ...]:
        with self._lock:
            ret = tuple(self._base_buffers.values())
        if order_by_access_time:
            ret = tuple(sorted(ret, key=lambda b: b.last_accessed))
        return ret

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
            unspilled = sum(
                buf.size for buf in self.base_buffers() if not buf.is_spilled
            )
            if unspilled < limit:
                break
            nbytes = self.spill_device_memory()
            if nbytes == 0:
                break  # No more to spill
            ret += nbytes
        return ret

    def lookup_address_range(
        self, ptr: int, size: int
    ) -> List[SpillableBuffer]:
        ret = []
        for buf in self.base_buffers():
            if buf.is_overlapping(ptr, size):
                ret.append(buf)
        return ret

    def __repr__(self) -> str:
        spilled = sum(
            buf.size for buf in self.base_buffers() if buf.is_spilled
        )
        unspilled = sum(
            buf.size for buf in self.base_buffers() if not buf.is_spilled
        )
        unspillable = 0
        for buf in self.base_buffers():
            if not (buf.is_spilled or buf.spillable):
                unspillable += buf.size
        unspillable_ratio = unspillable / unspilled if unspilled else 0

        return (
            f"<SpillManager spill_on_demand={self._spill_on_demand} "
            f"device_memory_limit={self._device_memory_limit} | "
            f"{format_bytes(spilled)} spilled | "
            f"{format_bytes(unspilled)} ({unspillable_ratio:.0%}) "
            f"unspilled (unspillable)>"
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


def _get_manager_from_env() -> Optional[SpillManager]:
    if not _env_get_bool("CUDF_SPILL", False):
        return None
    return SpillManager(
        spill_on_demand=_env_get_bool("CUDF_SPILL_ON_DEMAND", True),
        device_memory_limit=_env_get_int("CUDF_SPILL_DEVICE_LIMIT", None),
        statistic_level=_env_get_int("CUDF_SPILL_STATS_LEVEL", 0),
    )


# The global manager has three states:
#   - Uninitialized
#   - Initialized to None (spilling disabled)
#   - Initialized to a SpillManager instance (spilling enabled)
_global_manager_uninitialized: bool = True
_global_manager: Optional[SpillManager] = None


def global_manager_reset(manager: Optional[SpillManager]) -> None:
    """Set the global manager, which if None disables spilling"""

    global _global_manager, _global_manager_uninitialized
    if _global_manager is not None:
        gc.collect()
        base_buffers = _global_manager.base_buffers()
        if len(base_buffers) > 0:
            warnings.warn(f"overwriting non-empty manager: {base_buffers}")

    _global_manager = manager
    _global_manager_uninitialized = False


def global_manager_get() -> Optional[SpillManager]:
    """Get the global manager or None if spilling is disabled"""
    global _global_manager_uninitialized
    if _global_manager_uninitialized:
        global_manager_reset(_get_manager_from_env())
    return _global_manager
