# Copyright (c) 2022-2024, NVIDIA CORPORATION.

from __future__ import annotations

import gc
import io
import textwrap
import threading
import traceback
import warnings
import weakref
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import rmm.mr

from cudf.options import get_option
from cudf.utils.performance_tracking import _performance_tracking
from cudf.utils.string import format_bytes

if TYPE_CHECKING:
    from cudf.core.buffer.spillable_buffer import SpillableBufferOwner

_spill_cudf_nvtx_annotate = partial(
    _performance_tracking, domain="cudf_python-spill"
)


def get_traceback() -> str:
    """Pretty print current traceback to a string"""
    with io.StringIO() as f:
        traceback.print_stack(file=f)
        f.seek(0)
        return f.read()


def get_rmm_memory_resource_stack(
    mr: rmm.mr.DeviceMemoryResource,
) -> list[rmm.mr.DeviceMemoryResource]:
    """Get the RMM resource stack

    Parameters
    ----------
    mr : rmm.mr.DeviceMemoryResource
        Top of the resource stack

    Return
    ------
    list
        List of RMM resources
    """

    if hasattr(mr, "upstream_mr"):
        return [mr, *get_rmm_memory_resource_stack(mr.upstream_mr)]
    return [mr]


class SpillStatistics:
    """Gather spill statistics

    Levels of information gathered:
      0  - disabled (no overhead).
      1+ - duration and number of bytes spilled (very low overhead).
      2+ - a traceback for each time a spillable buffer is exposed
           permanently (potential high overhead).

    The statistics are printed when spilling-on-demand fails to find
    any buffer to spill. It is possible to retrieve the statistics
    manually through the spill manager, see example below.

    Parameters
    ----------
    level : int
        If not 0, enables statistics at the specified level.

    Examples
    --------
    >>> import cudf
    >>> from cudf.core.buffer.spill_manager import get_global_manager
    >>> manager = get_global_manager()
    >>> manager.statistics
    <SpillStatistics level=1>
    >>> df = cudf.DataFrame({"a": [1,2,3]})
    >>> manager.spill_to_device_limit(1)  # Spill df
    24
    >>> print(get_global_manager().statistics)
    Spill Statistics (level=1):
     Spilling (level >= 1):
      gpu => cpu: 24B in 0.0033579860000827466s
    """

    @dataclass
    class Expose:
        traceback: str
        count: int = 1
        total_nbytes: int = 0
        spilled_nbytes: int = 0

    spill_totals: dict[tuple[str, str], tuple[int, float]]

    def __init__(self, level) -> None:
        self.lock = threading.Lock()
        self.level = level
        self.spill_totals = defaultdict(lambda: (0, 0))
        # Maps each traceback to a Expose
        self.exposes: dict[str, SpillStatistics.Expose] = {}

    def log_spill(self, src: str, dst: str, nbytes: int, time: float) -> None:
        """Log a (un-)spilling event

        Parameters
        ----------
        src : str
            The memory location before spilling.
        dst : str
            The memory location after spilling.
        nbytes : int
            Number of bytes (un-)spilled.
        nbytes : float
            Elapsed time the event took in seconds.
        """
        if self.level < 1:
            return
        with self.lock:
            total_nbytes, total_time = self.spill_totals[(src, dst)]
            self.spill_totals[(src, dst)] = (
                total_nbytes + nbytes,
                total_time + time,
            )

    def log_expose(self, buf: SpillableBufferOwner) -> None:
        """Log an expose event

        We track logged exposes by grouping them by their traceback such
        that `self.exposes` maps tracebacks (as strings) to their logged
        data (as `Expose`).

        Parameters
        ----------
        buf : spillabe-buffer
            The buffer being exposed.
        """
        if self.level < 2:
            return
        with self.lock:
            tb = get_traceback()
            stat = self.exposes.get(tb, None)
            spilled_nbytes = buf.nbytes if buf.is_spilled else 0
            if stat is None:
                self.exposes[tb] = self.Expose(
                    traceback=tb,
                    total_nbytes=buf.nbytes,
                    spilled_nbytes=spilled_nbytes,
                )
            else:
                stat.count += 1
                stat.total_nbytes += buf.nbytes
                stat.spilled_nbytes += spilled_nbytes

    def __repr__(self) -> str:
        return f"<SpillStatistics level={self.level}>"

    def __str__(self) -> str:
        with self.lock:
            ret = f"Spill Statistics (level={self.level}):\n"
            if self.level == 0:
                return ret[:-1] + " N/A"

            # Print spilling stats
            ret += "  Spilling (level >= 1):"
            if len(self.spill_totals) == 0:
                ret += " None"
            ret += "\n"
            for (src, dst), (nbytes, time) in self.spill_totals.items():
                ret += f"    {src} => {dst}: "
                ret += f"{format_bytes(nbytes)} in {time:.3f}s\n"

            # Print expose stats
            ret += "  Exposed buffers (level >= 2): "
            if self.level < 2:
                return ret + "disabled"
            if len(self.exposes) == 0:
                ret += "None"
            ret += "\n"
            for s in sorted(self.exposes.values(), key=lambda x: -x.count):
                ret += textwrap.indent(
                    (
                        f"exposed {s.count} times, "
                        f"total: {format_bytes(s.total_nbytes)}, "
                        f"spilled: {format_bytes(s.spilled_nbytes)}, "
                        f"traceback:\n{s.traceback}"
                    ),
                    prefix=" " * 4,
                )
            return ret[:-1]  # Remove last `\n`


class SpillManager:
    """Manager of spillable buffers.

    This class implements tracking of all known spillable buffers, on-demand
    spilling of said buffers, and (optionally) maintains a memory usage limit.

    When `device_memory_limit=<limit-in-bytes>`, the manager will try keep
    the device memory usage below the specified limit by spilling of spillable
    buffers continuously, which will introduce a modest overhead.
    Notice, this is a soft limit. The memory usage might exceed the limit if
    too many buffers are unspillable.

    Parameters
    ----------
    device_memory_limit: int, optional
        If not None, this is the device memory limit in bytes that triggers
        device to host spilling. The global manager sets this to the value
        of `CUDF_SPILL_DEVICE_LIMIT` or None.
    statistic_level: int, optional
        If not 0, enables statistics at the specified level. See
        SpillStatistics for the different levels.
    """

    _buffers: weakref.WeakValueDictionary[int, SpillableBufferOwner]
    statistics: SpillStatistics

    def __init__(
        self,
        *,
        device_memory_limit: int | None = None,
        statistic_level: int = 0,
    ) -> None:
        self._lock = threading.Lock()
        self._buffers = weakref.WeakValueDictionary()
        self._id_counter = 0
        self._device_memory_limit = device_memory_limit
        self.statistics = SpillStatistics(statistic_level)

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
        # Let's try to spill device memory

        spilled = self.spill_device_memory(nbytes=nbytes)

        if spilled > 0:
            return True  # Ask RMM to retry the allocation

        if retry_once:
            # Let's collect garbage and try one more time
            gc.collect()
            return self._out_of_memory_handle(nbytes, retry_once=False)

        # TODO: write to log instead of stdout
        print(
            f"[WARNING] RMM allocation of {format_bytes(nbytes)} bytes "
            "failed, spill-on-demand couldn't find any device memory to "
            f"spill:\n{self!r}\ntraceback:\n{get_traceback()}\n"
            f"{self.statistics}"
        )
        return False  # Since we didn't find anything to spill, we give up

    def add(self, buffer: SpillableBufferOwner) -> None:
        """Add buffer to the set of managed buffers

        The manager keeps a weak reference to the buffer

        Parameters
        ----------
        buffer : SpillableBufferOwner
            The buffer to manage
        """
        if buffer.size > 0 and not buffer.exposed:
            with self._lock:
                self._buffers[self._id_counter] = buffer
                self._id_counter += 1
        self.spill_to_device_limit()

    def buffers(
        self, order_by_access_time: bool = False
    ) -> tuple[SpillableBufferOwner, ...]:
        """Get all managed buffers

        Parameters
        ----------
        order_by_access_time : bool, optional
            Order the buffer by access time (ascending order)

        Return
        ------
        tuple
            Tuple of buffers
        """
        with self._lock:
            ret = tuple(self._buffers.values())
        if order_by_access_time:
            ret = tuple(sorted(ret, key=lambda b: b.last_accessed))
        return ret

    @_spill_cudf_nvtx_annotate
    def spill_device_memory(self, nbytes: int) -> int:
        """Try to spill device memory

        This function is safe to call doing spill-on-demand
        since it does not lock buffers already locked.

        Parameters
        ----------
        nbytes : int
            Number of bytes to try to spill

        Return
        ------
        int
            Number of actually bytes spilled.
        """
        spilled = 0
        for buf in self.buffers(order_by_access_time=True):
            if buf.lock.acquire(blocking=False):
                try:
                    if not buf.is_spilled and buf.spillable:
                        buf.spill(target="cpu")
                        spilled += buf.size
                        if spilled >= nbytes:
                            break
                finally:
                    buf.lock.release()
        return spilled

    def spill_to_device_limit(self, device_limit: int | None = None) -> int:
        """Try to spill device memory until device limit

        Notice, by default this is a no-op.

        Parameters
        ----------
        device_limit : int, optional
            Limit in bytes. If None, the value of the environment variable
            `CUDF_SPILL_DEVICE_LIMIT` is used. If this is not set, the method
            does nothing and returns 0.

        Return
        ------
        int
            The number of bytes spilled.
        """
        limit = (
            self._device_memory_limit if device_limit is None else device_limit
        )
        if limit is None:
            return 0
        unspilled = sum(
            buf.size for buf in self.buffers() if not buf.is_spilled
        )
        return self.spill_device_memory(nbytes=unspilled - limit)

    def __repr__(self) -> str:
        spilled = sum(buf.size for buf in self.buffers() if buf.is_spilled)
        unspilled = sum(
            buf.size for buf in self.buffers() if not buf.is_spilled
        )
        unspillable = 0
        for buf in self.buffers():
            if not (buf.is_spilled or buf.spillable):
                unspillable += buf.size
        unspillable_ratio = unspillable / unspilled if unspilled else 0

        dev_limit = "N/A"
        if self._device_memory_limit is not None:
            dev_limit = format_bytes(self._device_memory_limit)

        return (
            f"<SpillManager device_memory_limit={dev_limit} | "
            f"{format_bytes(spilled)} spilled | "
            f"{format_bytes(unspilled)} ({unspillable_ratio:.0%}) "
            f"unspilled (unspillable)>"
        )


# The global manager has three states:
#   - Uninitialized
#   - Initialized to None (spilling disabled)
#   - Initialized to a SpillManager instance (spilling enabled)
_global_manager_uninitialized: bool = True
_global_manager: SpillManager | None = None


def set_global_manager(manager: SpillManager | None) -> None:
    """Set the global manager, which if None disables spilling"""

    global _global_manager, _global_manager_uninitialized
    if _global_manager is not None:
        gc.collect()
        buffers = _global_manager.buffers()
        if len(buffers) > 0:
            warnings.warn(f"overwriting non-empty manager: {buffers}")

    _global_manager = manager
    _global_manager_uninitialized = False


def get_global_manager() -> SpillManager | None:
    """Get the global manager or None if spilling is disabled"""
    global _global_manager_uninitialized
    if _global_manager_uninitialized:
        if get_option("spill"):
            manager = SpillManager(
                device_memory_limit=get_option("spill_device_limit"),
                statistic_level=get_option("spill_stats"),
            )
            set_global_manager(manager)
            if get_option("spill_on_demand"):
                set_spill_on_demand_globally()
        else:
            set_global_manager(None)
    return _global_manager


def set_spill_on_demand_globally() -> None:
    """Enable spill on demand in the current global spill manager.

    Warning: this modifies the current RMM memory resource. A memory resource
    to handle out-of-memory errors is pushed onto the RMM memory resource stack.

    Raises
    ------
    ValueError
        If no global spill manager exists (spilling is disabled).
    ValueError
        If a failure callback resource is already in the resource stack.
    """

    manager = get_global_manager()
    if manager is None:
        raise ValueError(
            "Cannot enable spill on demand with no global spill manager"
        )
    mr = rmm.mr.get_current_device_resource()
    if any(
        isinstance(m, rmm.mr.FailureCallbackResourceAdaptor)
        for m in get_rmm_memory_resource_stack(mr)
    ):
        raise ValueError(
            "Spill on demand (or another failure callback resource) "
            "is already registered"
        )
    rmm.mr.set_current_device_resource(
        rmm.mr.FailureCallbackResourceAdaptor(
            mr, manager._out_of_memory_handle
        )
    )


@contextmanager
def spill_on_demand_globally():
    """Context to enable spill on demand temporarily.

    Warning: this modifies the current RMM memory resource. A memory resource
    to handle out-of-memory errors is pushed onto the RMM memory resource stack
    when entering the context and popped again when exiting.

    Raises
    ------
    ValueError
        If no global spill manager exists (spilling is disabled).
    ValueError
        If a failure callback resource is already in the resource stack.
    ValueError
        If the RMM memory source stack was changed while in the context.
    """
    set_spill_on_demand_globally()
    # Save the new memory resource stack for later cleanup
    mr_stack = get_rmm_memory_resource_stack(
        rmm.mr.get_current_device_resource()
    )
    try:
        yield
    finally:
        mr = rmm.mr.get_current_device_resource()
        if mr_stack != get_rmm_memory_resource_stack(mr):
            raise ValueError(
                "RMM memory source stack was changed while in the context"
            )
        rmm.mr.set_current_device_resource(mr_stack[1])
