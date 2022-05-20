# Copyright (c) 2022, NVIDIA CORPORATION.

import os
import threading
import warnings
import weakref
from functools import cached_property
from typing import MutableMapping, Optional, Tuple

from cudf.core.buffer import Buffer


class SpillManager:
    def __init__(self, *, device_memory_limit=None) -> None:
        self._lock = threading.Lock()
        self._base_buffers: MutableMapping[
            int, Buffer
        ] = weakref.WeakValueDictionary()
        self._id_counter = 0
        self._device_memory_limit = device_memory_limit

    def __repr__(self) -> str:
        spilled, unspilled = self.spilled_and_unspilled()
        return (
            f"<SpillManager "
            f"device_memory_limit={self._device_memory_limit} "
            f"spilled: {spilled} unspilled: {unspilled}>"
        )

    def add(self, buffer: Buffer) -> None:
        with self._lock:
            if buffer.sole_owner:
                self._base_buffers[self._id_counter] = buffer
                self._id_counter += 1

    def base_buffers(self) -> Tuple[Buffer, ...]:
        with self._lock:
            return tuple(self._base_buffers.values())

    def spilled_and_unspilled(self) -> Tuple[int, int]:
        spilled, unspilled = 0, 0
        for buf in self.base_buffers():
            if buf.is_spilled:
                spilled += buf.size
            else:
                unspilled += buf.size
        return spilled, unspilled

    def spill_device_memory(self) -> int:
        # TODO: order spilling based on access time
        for buf in self.base_buffers():
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
            device_memory_limit=_env_get_int("CUDF_SPILL_DEVICE_LIMIT", None),
        )

    def clear(self) -> None:
        if self._manager is not None:
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

    def register_buffer(
        self,
        obj: Buffer,
    ) -> bool:
        if global_manager.enabled:
            global_manager.get().add(obj)
        return global_manager.enabled


global_manager = GlobalSpillManager()
