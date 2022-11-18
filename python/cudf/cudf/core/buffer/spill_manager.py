# Copyright (c) 2022, NVIDIA CORPORATION.

from __future__ import annotations

import gc
import io
import threading
import traceback
import warnings
import weakref
from typing import List, Optional, Tuple

import rmm.mr

from cudf.core.buffer.spillable_buffer import SpillableBuffer
from cudf.options import get_option
from cudf.utils.string import format_bytes


def get_traceback() -> str:
    """Pretty print current traceback to a string"""
    with io.StringIO() as f:
        traceback.print_stack(file=f)
        f.seek(0)
        return f.read()


def get_rmm_memory_resource_stack(
    mr: rmm.mr.DeviceMemoryResource,
) -> List[rmm.mr.DeviceMemoryResource]:
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
        return [mr] + get_rmm_memory_resource_stack(mr.upstream_mr)
    return [mr]


class SpillManager:
    """Manager of spillable buffers.

    This class implements tracking of all known spillable buffers, on-demand
    spilling of said buffers, and (optionally) maintains a memory usage limit.

    When `spill_on_demand=True`, the manager registers an RMM out-of-memory
    error handler, which will spill spillable buffers in order to free up
    memory.

    When `device_memory_limit=True`, the manager will try keep the device
    memory usage below the specified limit by spilling of spillable buffers
    continuously, which will introduce a modest overhead.

    Parameters
    ----------
    spill_on_demand : bool
        Enable spill on demand.
    device_memory_limit: int, optional
        If not None, this is the device memory limit in bytes that triggers
        device to host spilling. The global manager sets this to the value
        of `CUDF_SPILL_DEVICE_LIMIT` or None.
    """

    _buffers: weakref.WeakValueDictionary[int, SpillableBuffer]

    def __init__(
        self,
        *,
        spill_on_demand: bool = False,
        device_memory_limit: int = None,
    ) -> None:
        self._lock = threading.Lock()
        self._buffers = weakref.WeakValueDictionary()
        self._id_counter = 0
        self._spill_on_demand = spill_on_demand
        self._device_memory_limit = device_memory_limit

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
            f"spill:\n{repr(self)}\ntraceback:\n{get_traceback()}"
        )
        return False  # Since we didn't find anything to spill, we give up

    def add(self, buffer: SpillableBuffer) -> None:
        """Add buffer to the set of managed buffers

        The manager keeps a weak reference to the buffer

        Parameters
        ----------
        buffer : SpillableBuffer
            The buffer to manage
        """
        if buffer.size > 0 and not buffer.exposed:
            with self._lock:
                self._buffers[self._id_counter] = buffer
                self._id_counter += 1
        self.spill_to_device_limit()

    def buffers(
        self, order_by_access_time: bool = False
    ) -> Tuple[SpillableBuffer, ...]:
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

    def spill_to_device_limit(self, device_limit: int = None) -> int:
        """Spill until device limit

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
        ret = 0
        while True:
            unspilled = sum(
                buf.size for buf in self.buffers() if not buf.is_spilled
            )
            if unspilled < limit:
                break
            nbytes = self.spill_device_memory(nbytes=limit - unspilled)
            if nbytes == 0:
                break  # No more to spill
            ret += nbytes
        return ret

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

        return (
            f"<SpillManager spill_on_demand={self._spill_on_demand} "
            f"device_memory_limit={self._device_memory_limit} | "
            f"{format_bytes(spilled)} spilled | "
            f"{format_bytes(unspilled)} ({unspillable_ratio:.0%}) "
            f"unspilled (unspillable)>"
        )


# The global manager has three states:
#   - Uninitialized
#   - Initialized to None (spilling disabled)
#   - Initialized to a SpillManager instance (spilling enabled)
_global_manager_uninitialized: bool = True
_global_manager: Optional[SpillManager] = None


def set_global_manager(manager: Optional[SpillManager]) -> None:
    """Set the global manager, which if None disables spilling"""

    global _global_manager, _global_manager_uninitialized
    if _global_manager is not None:
        gc.collect()
        buffers = _global_manager.buffers()
        if len(buffers) > 0:
            warnings.warn(f"overwriting non-empty manager: {buffers}")

    _global_manager = manager
    _global_manager_uninitialized = False


def get_global_manager() -> Optional[SpillManager]:
    """Get the global manager or None if spilling is disabled"""
    global _global_manager_uninitialized
    if _global_manager_uninitialized:
        manager = None
        if get_option("spill"):
            manager = SpillManager(
                spill_on_demand=get_option("spill_on_demand"),
                device_memory_limit=get_option("spill_device_limit"),
            )
        set_global_manager(manager)
    return _global_manager
