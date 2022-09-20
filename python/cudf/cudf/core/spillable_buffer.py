# Copyright (c) 2022, NVIDIA CORPORATION.

from __future__ import annotations

import collections.abc
import pickle
import time
import weakref
from threading import RLock
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy

import rmm

from cudf.core.buffer import Buffer, Frame, get_ptr_and_size
from cudf.utils.string import format_bytes

if TYPE_CHECKING:
    from cudf.core.spill_manager import SpillManager


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
            return self._buf.ptr
        elif i == 1:
            return False
        raise IndexError("tuple index out of range")


class SpillableBuffer(Buffer):
    """A spillable buffer that implements DeviceBufferLike.

    This buffer supports spilling the represented data to host memory.
    Spilling can be done manually by calling `.__spill__(target="cpu")`
    but usually the associated spilling manager triggers spilling based on
    current device memory usage see `cudf.core.spill_manager.SpillManager`.
    Unspill is triggered automatically when accessing the data of the buffer.

    The buffer might not be spillable, which is based on the "expose" status
    of the buffer. We say that the buffer has been exposed if the device
    pointer (integer or void*) has been accessed outside of SpillableBuffer.
    In this case, we cannot invalidate the device pointer by moving the data
    to host.

    A buffer can be exposed permanently at creation or by accessing the `.ptr`
    property. To avoid this, one can use `.get_ptr()` instead, which support
    exposing the buffer temporarily.

    Parameters
    ----------
    data : buffer-like
        An buffer-like object representing device or host memory.
    exposed : bool, optional
        Whether or not a raw pointer (integer or C pointer) has
        been exposed to the outside world. If this is the case,
        the buffer cannot be spilled.
    manager : SpillManager
        The manager overseeing this buffer.
    """

    _ptr_desc: Dict[str, Any]
    _spill_locks: weakref.WeakSet

    def __init__(
        self,
        data: Any,
        exposed: bool,
        manager: SpillManager,
    ):
        self._lock = RLock()
        self._spill_locks = weakref.WeakSet()
        self._exposed = exposed
        self._last_accessed = time.monotonic()

        # First, we extract the memory pointer, size, and owner.
        # If it points to host memory we either:
        #   - copy to device memory if exposed=True
        #   - or create a new buffer that are marked as spilled already.
        if isinstance(data, SpillableBuffer):
            raise ValueError(
                "Cannot create from a SpillableBuffer, "
                "use __getitem__ to create a view"
            )
        if isinstance(data, rmm.DeviceBuffer):
            self._ptr_desc = {"type": "gpu"}
            self._ptr = data.ptr
            self._size = data.size
            self._owner = data
        elif hasattr(data, "__cuda_array_interface__"):
            self._ptr_desc = {"type": "gpu"}
            ptr, size = get_ptr_and_size(data.__cuda_array_interface__)
            self._ptr = ptr
            self._size = size
            self._owner = data
        else:
            if self._exposed:
                self._ptr_desc = {"type": "gpu"}
                ptr, size = get_ptr_and_size(
                    numpy.array(data).__array_interface__
                )
                buf = rmm.DeviceBuffer(ptr=ptr, size=size)
                self._ptr = buf.ptr
                self._size = buf.size
                self._owner = buf
                # Since we are copying the data, we know that the device
                # memory has not been exposed even if the original host
                # memory has been exposed.
                self._exposed = False
            else:
                data = memoryview(data)
                if not data.c_contiguous:
                    raise ValueError("memoryview must be C-contiguous")
                self._ptr_desc = {"type": "cpu", "memoryview": data}
                self._ptr = 0
                self._size = data.nbytes
                self._owner = None

        # Then, we inform the spilling manager about this new buffer if it is
        # not already known to the spilling manager.
        self._manager = manager
        base = self._manager.lookup_address_range(self._ptr, self._size)
        if base is not None:
            # Since this is a view, we expose the base buffer permanently by
            # accessing `.ptr`
            base.ptr
        elif self._exposed:
            # Since the buffer has been exposed permanently, we add it to
            # "others".
            self._manager.add_other(self)
        else:
            self._manager.add(self)

        # Finally, we spill until we comply with the device limit (if any).
        self._manager.spill_to_device_limit()

    @property
    def lock(self) -> RLock:
        return self._lock

    @property
    def is_spilled(self) -> bool:
        return self._ptr_desc["type"] != "gpu"

    def __spill__(self, target: str = "cpu") -> None:
        """Spill or un-spill this buffer in-place

        Parameters
        ----------
        target : str
            The target of the spilling.
        """

        with self._lock:
            ptr_type = self._ptr_desc["type"]
            if ptr_type == target:
                return

            if not self.spillable:
                raise ValueError(
                    f"Cannot in-place move an unspillable buffer: {self}"
                )

            if (ptr_type, target) == ("gpu", "cpu"):
                host_mem = memoryview(bytearray(self.size))
                rmm._lib.device_buffer.copy_ptr_to_host(self._ptr, host_mem)
                self._ptr_desc["memoryview"] = host_mem
                self._ptr = 0
                self._owner = None
            elif (ptr_type, target) == ("cpu", "gpu"):
                # Notice, this operation is prone to deadlock because the RMM
                # allocation might trigger spilling-on-demand which in turn
                # trigger a new call to this buffer's `__spill__()`.
                # Therefore, it is important that spilling-on-demand doesn't
                # tries to unspill an already locked buffer!
                dev_mem = rmm.DeviceBuffer.to_device(
                    self._ptr_desc.pop("memoryview")
                )
                self._ptr = dev_mem.ptr
                self._size = dev_mem.size
                self._owner = dev_mem
            else:
                # TODO: support moving to disk
                raise ValueError(f"Unknown target: {target}")
            self._ptr_desc["type"] = target

    @property
    def ptr(self) -> int:
        """Access the memory directly

        Notice, this will mark the buffer as "exposed" and make
        it unspillable permanently.

        Consider using `.get_ptr()` instead.
        """

        self._manager.spill_to_device_limit()
        with self._lock:
            if not self._exposed:
                self._manager.log_expose(self)
            self.__spill__(target="gpu")
            self._exposed = True
            self._last_accessed = time.monotonic()
            return self._ptr

    def spill_lock(self, spill_lock: SpillLock = None) -> SpillLock:
        if spill_lock is None:
            spill_lock = SpillLock()
        with self._lock:
            self.__spill__(target="gpu")
            self._spill_locks.add(spill_lock)
        return spill_lock

    def get_ptr(self, spill_lock: SpillLock = None) -> int:
        """Get a device pointer to the memory of the buffer

        If spill_lock is not None, a reference to this buffer is added
        to spill_lock, which disable spilling of this buffer while
        spill_lock is alive.

        Parameters
        ----------
        spill_lock : SpillLock, optional
            Adding a reference of this buffer to the spill lock.

        Return
        ------
        int
            The device pointer as an integer
        """

        if spill_lock is None:
            return self.ptr  # expose the buffer permanently

        self.spill_lock(spill_lock)
        self._last_accessed = time.monotonic()
        return self._ptr

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

    def memoryview(self, *, offset: int = 0, size: int = None) -> memoryview:
        size = self._size if size is None else size
        with self._lock:
            if self.spillable:
                self.__spill__(target="cpu")
                return self._ptr_desc["memoryview"][offset : offset + size]
            else:
                assert self._ptr_desc["type"] == "gpu"
                ret = memoryview(bytearray(size))
                rmm._lib.device_buffer.copy_ptr_to_host(
                    self._ptr + offset, ret
                )
                return ret

    def _getitem(self, offset: int, size: int) -> Buffer:
        return SpillableBufferView(base=self, offset=offset, size=size)

    def serialize(self) -> Tuple[dict, List[Frame]]:
        header: Dict[Any, Any]
        frames: List[Frame]
        with self._lock:
            header = {}
            header["type-serialized"] = pickle.dumps(self.__class__)
            header["frame_count"] = 1
            if self.is_spilled:
                frames = [self.memoryview()]
            else:
                spill_lock = SpillLock()
                ptr = self.get_ptr(spill_lock=spill_lock)
                frames = [
                    Buffer(
                        data=ptr,
                        size=self.size,
                        owner=(self._owner, spill_lock),
                    )
                ]
            return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> SpillableBuffer:
        from cudf.core.spill_manager import global_manager

        if header["frame_count"] != 1:
            raise ValueError(
                "Deserializing a SpillableBuffer expect a single frame"
            )
        (frame,) = frames
        if isinstance(frame, SpillableBuffer):
            ret = frame
        else:
            ret = cls(frame, exposed=False, manager=global_manager.get())
        return ret

    def is_overlapping(self, ptr: int, size: int):
        with self._lock:
            return (
                not self.is_spilled
                and (ptr + size) > self._ptr
                and (self._ptr + self._size) > ptr
            )

    def __repr__(self) -> str:
        if self._ptr_desc["type"] != "gpu":
            ptr_info = str(self._ptr_desc)
        else:
            ptr_info = str(hex(self._ptr))
        return (
            f"<SpillableBuffer size={format_bytes(self._size)} "
            f"spillable={self.spillable} exposed={self.exposed} "
            f"num-spill-locks={len(self._spill_locks)} "
            f"ptr={ptr_info} owner={repr(self._owner)}"
        )


class SpillableBufferView(SpillableBuffer):
    """A view of a spillable buffer

    This buffer deligates must operations to its base buffer.

    Parameters
    ----------
    base : SpillableBuffer
        The base of the view
    offset : int
        Memory offset into the base buffer
    size : int
        Size of the view (in bytes)
    """

    def __init__(self, base: SpillableBuffer, offset: int, size: int) -> None:
        if size < 0:
            raise ValueError("size cannot be negative")
        if offset < 0:
            raise ValueError("offset cannot be negative")
        if offset + size > base.size:
            raise ValueError(
                "offset+size cannot be greater than the size of base"
            )
        self._base = base
        self._offset = offset
        self._size = size
        self._owner = base
        self._lock = base.lock

    @property
    def ptr(self) -> int:
        return self._base.ptr + self._offset

    def get_ptr(self, spill_lock: SpillLock = None) -> int:
        return self._base.get_ptr(spill_lock=spill_lock) + self._offset

    def _getitem(self, offset: int, size: int) -> Buffer:
        return SpillableBufferView(
            base=self._base, offset=offset + self._offset, size=size
        )

    def memoryview(self, *, offset: int = 0, size: int = None) -> memoryview:
        return self._base.memoryview(offset=self._offset + offset, size=size)

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> SpillableBuffer:
        return SpillableBuffer.deserialize(header, frames)

    def __repr__(self) -> str:
        return (
            f"<SpillableBufferView size={format_bytes(self._size)} "
            f"offset={format_bytes(self._offset)} of {self._base} "
        )

    # The rest of the methods deligate to the base buffer.
    def __spill__(self, target: str = "cpu") -> None:
        return self._base.__spill__(target=target)

    @property
    def is_spilled(self) -> bool:
        return self._base.is_spilled

    @property
    def exposed(self) -> bool:
        return self._base.exposed

    @property
    def spillable(self) -> bool:
        return self._base.spillable

    def spill_lock(self, spill_lock: SpillLock = None) -> SpillLock:
        return self._base.spill_lock(spill_lock=spill_lock)
