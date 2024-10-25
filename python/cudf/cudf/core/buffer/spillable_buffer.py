# Copyright (c) 2022-2024, NVIDIA CORPORATION.

from __future__ import annotations

import collections.abc
import pickle
import time
import weakref
from threading import RLock
from typing import TYPE_CHECKING, Any, Literal

import numpy
import nvtx
from typing_extensions import Self

import rmm

from cudf.core.buffer.buffer import (
    Buffer,
    BufferOwner,
    cuda_array_interface_wrapper,
    host_memory_allocation,
)
from cudf.core.buffer.exposure_tracked_buffer import ExposureTrackedBuffer
from cudf.utils.performance_tracking import _get_color_for_nvtx
from cudf.utils.string import format_bytes

if TYPE_CHECKING:
    from cudf.core.buffer.spill_manager import SpillManager


class SpillLock:
    pass


class DelayedPointerTuple(collections.abc.Sequence):
    """
    A delayed version of the "data" field in __cuda_array_interface__.

    The idea is to delay the access to `Buffer.ptr` until the user
    actually accesses the data pointer.

    For instance, in many cases __cuda_array_interface__ is accessed
    only to determine whether an object is a CUDA object or not.

    TODO: this doesn't support libraries such as PyTorch that declare
    the tuple of __cuda_array_interface__["data"] in Cython. In such
    cases, Cython will raise an error because DelayedPointerTuple
    isn't a "real" tuple.
    """

    def __init__(self, buffer) -> None:
        self._buf = buffer

    def __len__(self):
        return 2

    def __getitem__(self, i):
        if i == 0:
            return self._buf.get_ptr(mode="write")
        elif i == 1:
            return False
        raise IndexError("tuple index out of range")


class SpillableBufferOwner(BufferOwner):
    """A Buffer that supports spilling memory off the GPU to avoid OOMs.

    This buffer supports spilling the represented data to host memory.
    Spilling can be done manually by calling `.spill(target="cpu")` but
    usually the associated spilling manager triggers spilling based on current
    device memory usage see `cudf.core.buffer.spill_manager.SpillManager`.
    Unspill is triggered automatically when accessing the data of the buffer.

    The buffer might not be spillable, which is based on the "expose" status of
    the buffer. We say that the buffer has been exposed if the device pointer
    (integer or void*) has been accessed outside of SpillableBufferOwner.
    In this case, we cannot invalidate the device pointer by moving the data
    to host.

    A buffer can be exposed permanently at creation or by accessing the `.ptr`
    property. To avoid this, one can use `.get_ptr()` instead, which support
    exposing the buffer temporarily.

    Use the factory function `as_buffer` to create a SpillableBufferOwner
    instance.
    """

    lock: RLock
    _spill_locks: weakref.WeakSet
    _last_accessed: float
    _ptr_desc: dict[str, Any]
    _manager: SpillManager

    def _finalize_init(self, ptr_desc: dict[str, Any]) -> None:
        """Finish initialization of the spillable buffer

        This implements the common initialization that `from_device_memory`
        and `from_host_memory` are missing.

        Parameters
        ----------
        ptr_desc : dict
            Description of the memory.
        """

        from cudf.core.buffer.spill_manager import get_global_manager

        self.lock = RLock()
        self._spill_locks = weakref.WeakSet()
        self._last_accessed = time.monotonic()
        self._ptr_desc = ptr_desc
        manager = get_global_manager()
        if manager is None:
            raise ValueError(
                f"cannot create {self.__class__} without "
                "a global spill manager"
            )

        self._manager = manager
        self._manager.add(self)

    @classmethod
    def from_device_memory(cls, data: Any, exposed: bool) -> Self:
        """Create a spillabe buffer from device memory.

        No data is being copied.

        Parameters
        ----------
        data : device-buffer-like
            An object implementing the CUDA Array Interface.
        exposed : bool
            Mark the buffer as permanently exposed (unspillable).

        Returns
        -------
        SpillableBufferOwner
            Buffer representing the same device memory as `data`
        """
        ret = super().from_device_memory(data, exposed=exposed)
        ret._finalize_init(ptr_desc={"type": "gpu"})
        return ret

    @classmethod
    def from_host_memory(cls, data: Any) -> Self:
        """Create a spillabe buffer from host memory.

        Data must implement `__array_interface__`, the buffer protocol, and/or
        be convertible to a buffer object using `numpy.asanyarray()`

        The new buffer is marked as spilled to host memory already.

        Raises ValueError if array isn't C-contiguous.

        Parameters
        ----------
        data : Any
            An object that represents host memory.

        Returns
        -------
        SpillableBufferOwner
            Buffer representing a copy of `data`.
        """

        # Convert to a memoryview using numpy array, this will not copy data
        # in most cases.
        data = memoryview(numpy.asanyarray(data))
        if not data.c_contiguous:
            raise ValueError("Buffer data must be C-contiguous")
        data = data.cast("B")  # Make sure itemsize==1

        # Create an already spilled buffer
        ret = cls(ptr=0, size=data.nbytes, owner=None, exposed=False)
        ret._finalize_init(ptr_desc={"type": "cpu", "memoryview": data})
        return ret

    @property
    def is_spilled(self) -> bool:
        return self._ptr_desc["type"] != "gpu"

    def spill(self, target: str = "cpu") -> None:
        """Spill or un-spill this buffer in-place

        Parameters
        ----------
        target : str
            The target of the spilling.
        """

        time_start = time.perf_counter()
        with self.lock:
            ptr_type = self._ptr_desc["type"]
            if ptr_type == target:
                return

            if not self.spillable:
                raise ValueError(
                    f"Cannot in-place move an unspillable buffer: {self}"
                )

            if (ptr_type, target) == ("gpu", "cpu"):
                with nvtx.annotate(
                    message="SpillDtoH",
                    color=_get_color_for_nvtx("SpillDtoH"),
                    domain="cudf_python-spill",
                ):
                    host_mem = host_memory_allocation(self.size)
                    rmm.pylibrmm.device_buffer.copy_ptr_to_host(
                        self._ptr, host_mem
                    )
                self._ptr_desc["memoryview"] = host_mem
                self._ptr = 0
                self._owner = None
            elif (ptr_type, target) == ("cpu", "gpu"):
                # Notice, this operation is prone to deadlock because the RMM
                # allocation might trigger spilling-on-demand which in turn
                # trigger a new call to this buffer's `spill()`.
                # Therefore, it is important that spilling-on-demand doesn't
                # try to unspill an already locked buffer!
                with nvtx.annotate(
                    message="SpillHtoD",
                    color=_get_color_for_nvtx("SpillHtoD"),
                    domain="cudf_python-spill",
                ):
                    dev_mem = rmm.DeviceBuffer.to_device(
                        self._ptr_desc.pop("memoryview")
                    )
                self._ptr = dev_mem.ptr
                self._owner = dev_mem
                assert self._size == dev_mem.size
            else:
                # TODO: support moving to disk
                raise ValueError(f"Unknown target: {target}")
            self._ptr_desc["type"] = target

        time_end = time.perf_counter()
        self._manager.statistics.log_spill(
            src=ptr_type,
            dst=target,
            nbytes=self.size,
            time=time_end - time_start,
        )

    def mark_exposed(self) -> None:
        """Mark the buffer as "exposed" and make it unspillable permanently.

        This also unspills the buffer (unspillable buffers cannot be spilled!).
        """

        self._manager.spill_to_device_limit()
        with self.lock:
            if not self.exposed:
                self._manager.statistics.log_expose(self)
            self.spill(target="gpu")
            super().mark_exposed()
            self._last_accessed = time.monotonic()

    def spill_lock(self, spill_lock: SpillLock) -> None:
        """Spill lock the buffer

        Mark the buffer as unspillable while `spill_lock` is alive,
        which is tracked by monitoring a weakref to `spill_lock`.

        Parameters
        ----------
        spill_lock : SpillLock
            The object that defines the scope of the lock.
        """

        with self.lock:
            self.spill(target="gpu")
            self._spill_locks.add(spill_lock)

    def get_ptr(self, *, mode: Literal["read", "write"]) -> int:
        """Get a device pointer to the memory of the buffer.

        If this is called within an `acquire_spill_lock` context,
        a reference to this buffer is added to spill_lock, which
        disable spilling of this buffer while in the context.

        If this is *not* called within a `acquire_spill_lock` context,
        this buffer is marked as unspillable permanently.

        Returns
        -------
        int
            The device pointer as an integer
        """
        from cudf.core.buffer.utils import get_spill_lock

        spill_lock = get_spill_lock()
        if spill_lock is None:
            self.mark_exposed()
        else:
            self.spill_lock(spill_lock)
            self._last_accessed = time.monotonic()
        return self._ptr

    def memory_info(self) -> tuple[int, int, str]:
        """Get pointer, size, and device type of this buffer.

        Warning, it is not safe to access the pointer value without
        spill lock the buffer manually. This method neither exposes
        nor spill locks the buffer.

        Return
        ------
        int
            The memory pointer as an integer (device or host memory)
        int
            The size of the memory in bytes
        str
            The device type as a string ("cpu" or "gpu")
        """

        if self._ptr_desc["type"] == "gpu":
            ptr = self._ptr
        elif self._ptr_desc["type"] == "cpu":
            ptr = numpy.array(
                self._ptr_desc["memoryview"], copy=False
            ).__array_interface__["data"][0]
        return (ptr, self.nbytes, self._ptr_desc["type"])

    @property
    def spillable(self) -> bool:
        return not self.exposed and len(self._spill_locks) == 0

    @property
    def last_accessed(self) -> float:
        return self._last_accessed

    @property
    def __cuda_array_interface__(self) -> dict:
        return {
            "data": DelayedPointerTuple(self),
            "shape": (self.size,),
            "strides": None,
            "typestr": "|u1",
            "version": 0,
        }

    def memoryview(
        self, *, offset: int = 0, size: int | None = None
    ) -> memoryview:
        size = self._size if size is None else size
        with self.lock:
            if self.spillable:
                self.spill(target="cpu")
                return self._ptr_desc["memoryview"][offset : offset + size]
            else:
                assert self._ptr_desc["type"] == "gpu"
                ret = host_memory_allocation(size)
                rmm.pylibrmm.device_buffer.copy_ptr_to_host(
                    self._ptr + offset, ret
                )
                return ret

    def __str__(self) -> str:
        if self._ptr_desc["type"] != "gpu":
            ptr_info = str(self._ptr_desc)
        else:
            ptr_info = str(hex(self._ptr))
        return (
            f"<{self.__class__.__name__} size={format_bytes(self._size)} "
            f"spillable={self.spillable} exposed={self.exposed} "
            f"num-spill-locks={len(self._spill_locks)} "
            f"ptr={ptr_info} owner={repr(self._owner)}>"
        )


class SpillableBuffer(ExposureTrackedBuffer):
    """A slice of a spillable buffer"""

    _owner: SpillableBufferOwner

    def spill(self, target: str = "cpu") -> None:
        return self._owner.spill(target=target)

    @property
    def is_spilled(self) -> bool:
        return self._owner.is_spilled

    @property
    def spillable(self) -> bool:
        return self._owner.spillable

    def spill_lock(self, spill_lock: SpillLock) -> None:
        self._owner.spill_lock(spill_lock=spill_lock)

    def memory_info(self) -> tuple[int, int, str]:
        (ptr, _, device_type) = self._owner.memory_info()
        return (ptr + self._offset, self.nbytes, device_type)

    def serialize(self) -> tuple[dict, list]:
        """Serialize the Buffer

        Normally, we would use `[self]` as the frames. This would work but
        also mean that `self` becomes exposed permanently if the frames are
        later accessed through `__cuda_array_interface__`, which is exactly
        what libraries like Dask+UCX would do when communicating!

        The sound solution is to modify Dask et al. so that they access the
        frames through `.get_ptr()` and holds on to the `spill_lock` until
        the frame has been transferred. However, until this adaptation we
        use a hack where the frame is a `Buffer` with a `spill_lock` as the
        owner, which makes `self` unspillable while the frame is alive but
        doesn't expose `self` when `__cuda_array_interface__` is accessed.

        Warning, this hack means that the returned frame must be copied before
        given to `.deserialize()`, otherwise we would have a `Buffer` pointing
        to memory already owned by an existing `SpillableBufferOwner`.
        """
        header: dict[str, Any] = {}
        frames: list[Buffer | memoryview]
        with self._owner.lock:
            header["type-serialized"] = pickle.dumps(self.__class__)
            header["owner-type-serialized"] = pickle.dumps(type(self._owner))
            header["frame_count"] = 1
            if self.is_spilled:
                frames = [self.memoryview()]
            else:
                # TODO: Use `frames=[self]` instead of this hack, see doc above
                spill_lock = SpillLock()
                self.spill_lock(spill_lock)
                ptr, size, _ = self.memory_info()
                frames = [
                    Buffer(
                        owner=BufferOwner.from_device_memory(
                            cuda_array_interface_wrapper(
                                ptr=ptr,
                                size=size,
                                owner=(self._owner, spill_lock),
                            ),
                            exposed=False,
                        )
                    )
                ]
            return header, frames

    def copy(self, deep: bool = True) -> Self:
        from cudf.core.buffer.utils import acquire_spill_lock

        if not deep:
            return super().copy(deep=False)

        if self.is_spilled:
            # In this case, we make the new copy point to the same spilled
            # data in host memory. We can do this since spilled data is never
            # modified.
            owner = self._owner.from_host_memory(self.memoryview())
            return self.__class__(owner=owner, offset=0, size=owner.size)

        with acquire_spill_lock():
            return super().copy(deep=deep)

    @property
    def __cuda_array_interface__(self) -> dict:
        return {
            "data": DelayedPointerTuple(self),
            "shape": (self.size,),
            "strides": None,
            "typestr": "|u1",
            "version": 0,
        }
