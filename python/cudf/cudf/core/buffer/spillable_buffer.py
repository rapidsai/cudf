# Copyright (c) 2022-2023, NVIDIA CORPORATION.

from __future__ import annotations

import collections.abc
import pickle
import time
import weakref
from threading import RLock
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

import numpy
from typing_extensions import Self

import rmm

from cudf.core.buffer.buffer import (
    Buffer,
    BufferOwner,
    cuda_array_interface_wrapper,
    get_owner,
    get_ptr_and_size,
    host_memory_allocation,
)
from cudf.utils.string import format_bytes

if TYPE_CHECKING:
    from cudf.core.buffer.spill_manager import SpillManager


def as_spillable_buffer(data, exposed: bool) -> SpillableBufferSlice:
    """Factory function to wrap `data` in a SpillableBufferSlice object.

    If `data` isn't a buffer already, a new buffer that points to the memory of
    `data` is created. If `data` represents host memory, it is copied to a new
    `rmm.DeviceBuffer` device allocation. Otherwise, the memory of `data` is
    **not** copied, instead the new buffer keeps a reference to `data` in order
    to retain its lifetime.

    If `data` is owned by a spillable buffer, a "slice" of the buffer is
    returned. In this case, the spillable buffer must either be "exposed" or
    spilled locked (called within an acquire_spill_lock context). This is to
    guarantee that the memory of `data` isn't spilled before this function gets
    to calculate the offset of the new slice.

    It is illegal for a spillable buffer to own another spillable buffer.

    Parameters
    ----------
    data : buffer-like or array-like
        A buffer-like or array-like object that represent C-contiguous memory.
    exposed : bool, optional
        Mark the buffer as permanently exposed (unspillable).

    Return
    ------
    SpillableBufferSlice
        A spillabe buffer instance that represents the device memory of `data`.
    """

    from cudf.core.buffer.utils import get_spill_lock

    if not hasattr(data, "__cuda_array_interface__"):
        if exposed:
            raise ValueError("cannot created exposed host memory")
        tracked_buf = SpillableBuffer._from_host_memory(data)
        return SpillableBufferSlice(
            owner=tracked_buf, offset=0, size=tracked_buf.size
        )

    if not hasattr(data, "__cuda_array_interface__"):
        if exposed:
            raise ValueError("cannot created exposed host memory")
        spillabe_buf = SpillableBuffer._from_host_memory(data)
        return SpillableBufferSlice(
            owner=spillabe_buf, offset=0, size=spillabe_buf.size
        )

    owner = get_owner(data, SpillableBuffer)
    if owner is not None:
        if not owner.exposed and get_spill_lock() is None:
            raise ValueError(
                "A owning spillable buffer must "
                "either be exposed or spilled locked."
            )
        # `data` is owned by an spillable buffer, which is exposed or
        # spilled locked.
        ptr, size = get_ptr_and_size(data.__cuda_array_interface__)
        base_ptr = owner.get_ptr(mode="read")
        if size > 0 and owner._ptr == 0:
            raise ValueError(
                "Cannot create a non-empty slice of a null buffer"
            )
        return SpillableBufferSlice(
            owner=owner, offset=ptr - base_ptr, size=size
        )

    # `data` is new device memory
    owner = SpillableBuffer._from_device_memory(data, exposed=exposed)
    return SpillableBufferSlice(owner=owner, offset=0, size=owner.size)


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


class SpillableBuffer(BufferOwner):
    """A Buffer that supports spilling memory off the GPU to avoid OOMs.

    This buffer supports spilling the represented data to host memory.
    Spilling can be done manually by calling `.spill(target="cpu")` but
    usually the associated spilling manager triggers spilling based on current
    device memory usage see `cudf.core.buffer.spill_manager.SpillManager`.
    Unspill is triggered automatically when accessing the data of the buffer.

    The buffer might not be spillable, which is based on the "expose" status
    of the buffer. We say that the buffer has been exposed if the device
    pointer (integer or void*) has been accessed outside of SpillableBuffer.
    In this case, we cannot invalidate the device pointer by moving the data
    to host.

    A buffer can be exposed permanently at creation or by accessing the `.ptr`
    property. To avoid this, one can use `.get_ptr()` instead, which support
    exposing the buffer temporarily.

    Use the factory function `as_buffer` to create a SpillableBuffer instance.
    """

    lock: RLock
    _spill_locks: weakref.WeakSet
    _last_accessed: float
    _ptr_desc: Dict[str, Any]
    _exposed: bool
    _manager: SpillManager

    def _finalize_init(self, ptr_desc: Dict[str, Any], exposed: bool) -> None:
        """Finish initialization of the spillable buffer

        This implements the common initialization that `_from_device_memory`
        and `_from_host_memory` are missing.

        Parameters
        ----------
        ptr_desc : dict
            Description of the memory.
        exposed : bool, optional
            Mark the buffer as permanently exposed (unspillable).
        """

        from cudf.core.buffer.spill_manager import get_global_manager

        self.lock = RLock()
        self._spill_locks = weakref.WeakSet()
        self._last_accessed = time.monotonic()
        self._ptr_desc = ptr_desc
        self._exposed = exposed
        manager = get_global_manager()
        if manager is None:
            raise ValueError(
                f"cannot create {self.__class__} without "
                "a global spill manager"
            )

        self._manager = manager
        self._manager.add(self)

    @classmethod
    def _from_device_memory(cls, data: Any, *, exposed: bool = False) -> Self:
        """Create a spillabe buffer from device memory.

        No data is being copied.

        Parameters
        ----------
        data : device-buffer-like
            An object implementing the CUDA Array Interface.
        exposed : bool, optional
            Mark the buffer as permanently exposed (unspillable).

        Returns
        -------
        SpillableBuffer
            Buffer representing the same device memory as `data`
        """
        ret = super()._from_device_memory(data)
        ret._finalize_init(ptr_desc={"type": "gpu"}, exposed=exposed)
        return ret

    @classmethod
    def _from_host_memory(cls, data: Any) -> Self:
        """Create a spillabe buffer from host memory.

        Data must implement `__array_interface__`, the buffer protocol, and/or
        be convertible to a buffer object using `numpy.array()`

        The new buffer is marked as spilled to host memory already.

        Raises ValueError if array isn't C-contiguous.

        Parameters
        ----------
        data : Any
            An object that represens host memory.

        Returns
        -------
        SpillableBuffer
            Buffer representing a copy of `data`.
        """

        # Convert to a memoryview using numpy array, this will not copy data
        # in most cases.
        data = memoryview(numpy.array(data, copy=False, subok=True))
        if not data.c_contiguous:
            raise ValueError("Buffer data must be C-contiguous")
        data = data.cast("B")  # Make sure itemsize==1

        # Create an already spilled buffer
        ret = cls.__new__(cls)
        ret._owner = None
        ret._ptr = 0
        ret._size = data.nbytes
        ret._finalize_init(
            ptr_desc={"type": "cpu", "memoryview": data}, exposed=False
        )
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
                host_mem = host_memory_allocation(self.size)
                rmm._lib.device_buffer.copy_ptr_to_host(self._ptr, host_mem)
                self._ptr_desc["memoryview"] = host_mem
                self._ptr = 0
                self._owner = None
            elif (ptr_type, target) == ("cpu", "gpu"):
                # Notice, this operation is prone to deadlock because the RMM
                # allocation might trigger spilling-on-demand which in turn
                # trigger a new call to this buffer's `spill()`.
                # Therefore, it is important that spilling-on-demand doesn't
                # try to unspill an already locked buffer!
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
            if not self._exposed:
                self._manager.statistics.log_expose(self)
            self.spill(target="gpu")
            self._exposed = True
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

    def memory_info(self) -> Tuple[int, int, str]:
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
    def owner(self) -> Any:
        return self._owner

    @property
    def exposed(self) -> bool:
        return self._exposed

    @property
    def spillable(self) -> bool:
        return not self._exposed and len(self._spill_locks) == 0

    @property
    def size(self) -> int:
        return self._size

    @property
    def nbytes(self) -> int:
        return self._size

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
        self, *, offset: int = 0, size: Optional[int] = None
    ) -> memoryview:
        size = self._size if size is None else size
        with self.lock:
            if self.spillable:
                self.spill(target="cpu")
                return self._ptr_desc["memoryview"][offset : offset + size]
            else:
                assert self._ptr_desc["type"] == "gpu"
                ret = host_memory_allocation(size)
                rmm._lib.device_buffer.copy_ptr_to_host(
                    self._ptr + offset, ret
                )
                return ret

    def __repr__(self) -> str:
        if self._ptr_desc["type"] != "gpu":
            ptr_info = str(self._ptr_desc)
        else:
            ptr_info = str(hex(self._ptr))
        return (
            f"<SpillableBuffer size={format_bytes(self._size)} "
            f"spillable={self.spillable} exposed={self.exposed} "
            f"num-spill-locks={len(self._spill_locks)} "
            f"ptr={ptr_info} owner={repr(self._owner)}>"
        )


class SpillableBufferSlice(Buffer):
    """A slice of a spillable buffer

    This buffer applies the slicing and then delegates all
    operations to its owning buffer.

    Parameters
    ----------
    owner : SpillableBuffer
        The owner of the view
    offset : int
        Memory offset into the owning buffer
    size : int
        Size of the view (in bytes)
    """

    _owner: SpillableBuffer

    def __init__(self, owner: SpillableBuffer, offset: int, size: int) -> None:
        if size < 0:
            raise ValueError("size cannot be negative")
        if offset < 0:
            raise ValueError("offset cannot be negative")
        if offset + size > owner.size:
            raise ValueError(
                "offset+size cannot be greater than the size of the owner"
            )
        self._owner = owner
        self._offset = offset
        self._size = size
        self.lock = owner.lock

    def spill(self, target: str = "cpu") -> None:
        return self._owner.spill(target=target)

    @property
    def is_spilled(self) -> bool:
        return self._owner.is_spilled

    @property
    def exposed(self) -> bool:
        return self._owner.exposed

    @property
    def spillable(self) -> bool:
        return self._owner.spillable

    def spill_lock(self, spill_lock: SpillLock) -> None:
        self._owner.spill_lock(spill_lock=spill_lock)

    def memory_info(self) -> Tuple[int, int, str]:
        (ptr, _, device_type) = self._owner.memory_info()
        return (ptr + self._offset, self.nbytes, device_type)

    def mark_exposed(self) -> None:
        self._owner.mark_exposed()

    def serialize(self) -> Tuple[dict, list]:
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
        to memory already owned by an existing `SpillableBuffer`.
        """
        header: Dict[str, Any] = {}
        frames: List[BufferOwner | memoryview]
        with self.lock:
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
                    BufferOwner._from_device_memory(
                        cuda_array_interface_wrapper(
                            ptr=ptr,
                            size=size,
                            owner=(self._owner, spill_lock),
                        )
                    )
                ]
            return header, frames

    @property
    def __cuda_array_interface__(self) -> dict:
        return {
            "data": DelayedPointerTuple(self),
            "shape": (self.size,),
            "strides": None,
            "typestr": "|u1",
            "version": 0,
        }
