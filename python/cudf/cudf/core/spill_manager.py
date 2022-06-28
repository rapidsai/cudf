# Copyright (c) 2022, NVIDIA CORPORATION.

import gc
import io
import os
import threading
import traceback
import warnings
import weakref
from functools import cached_property
from typing import MutableMapping, Optional, Tuple

import rmm.mr

from cudf.core.buffer import Buffer, format_bytes


class SpillManager:
    def __init__(
        self, *, spill_on_demand=False, device_memory_limit=None
    ) -> None:
        self._lock = threading.Lock()
        self._base_buffers: MutableMapping[
            int, Buffer
        ] = weakref.WeakValueDictionary()
        self._id_counter = 0
        self._spill_on_demand = spill_on_demand
        self._device_memory_limit = device_memory_limit
        if self._spill_on_demand:
            self.register_spill_on_demand()

    def register_spill_on_demand(self):
        # TODO: check if a `FailureCallbackResourceAdaptor` has been
        #       registered already
        def oom(nbytes: int) -> bool:
            """Try to handle an out-of-memory error by spilling"""

            # Keep spilling until `nbytes` been spilled
            total_spilled = 0
            while total_spilled < nbytes:
                spilled = self.spill_device_memory()
                if spilled == 0:
                    break  # No more to spill!
                total_spilled += spilled

            # cuDF has a lot of circular references
            gc.collect()

            if total_spilled > 0:
                return True  # Ask RMM to retry the allocation

            with io.StringIO() as f:
                traceback.print_stack(file=f)
                f.seek(0)
                tb = f.read()
            # TODO: write to log instead of stdout
            print(
                f"[WARNING] RMM allocation of {nbytes} bytes failed, "
                "spill-on-demand couldn't find any device memory to "
                f"spill:\n{repr(self)}\ntraceback:\n{tb}\n"
            )
            return False  # Since we didn't find anything to spill, we give up

        current_mr = rmm.mr.get_current_device_resource()
        mr = rmm.mr.FailureCallbackResourceAdaptor(current_mr, oom)
        rmm.mr.set_current_device_resource(mr)

    def add(self, buffer: Buffer) -> None:
        with self._lock:
            if buffer.size > 0 and not buffer.ptr_exposed:
                self._base_buffers[self._id_counter] = buffer
                self._id_counter += 1
        self.spill_to_device_limit()

    def base_buffers(
        self, order_by_access_time: bool = False
    ) -> Tuple[Buffer, ...]:
        with self._lock:
            ret = tuple(self._base_buffers.values())
        if order_by_access_time:
            ret = tuple(sorted(ret, key=lambda b: b.last_accessed))
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
        for buf in self.base_buffers(order_by_access_time=True):
            with buf._lock:
                if not buf.is_spilled and buf.spillable:
                    buf.move_inplace(target="cpu")
                    return buf.size
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

    def lookup_address_range(self, ptr: int, size: int) -> Optional[Buffer]:
        end = ptr + size
        hit: Optional[Buffer] = None
        for buf in self.base_buffers():
            if not buf.is_spilled and buf._ptr and buf.size:
                buf_end = buf._ptr + buf.size
                if end > buf._ptr and buf_end > ptr:
                    if hit is not None:
                        raise RuntimeError(
                            f"Two base buffers overlap: {hit} and {buf}"
                        )
                    hit = buf
        return hit

    def __repr__(self) -> str:
        spilled, unspilled = self.spilled_and_unspilled()
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
            "unspilled (unspillable)>"
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
