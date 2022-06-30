# Copyright (c) 2020-2022, NVIDIA CORPORATION.
from __future__ import annotations

import functools
import operator
import pickle
import time
from threading import RLock
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

import numpy as np

import rmm

import cudf
from cudf.core.abc import Serializable

if TYPE_CHECKING:
    from cudf._lib.column import AccessCounter
    from cudf.core.spill_manager import SpillManager


def format_bytes(nbytes: int) -> str:
    n = float(nbytes)
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if abs(n) < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024
    return f"{n:.2f} TiB"


class DelayedPointerTuple(Sequence):
    """
    A delayed version of the "data" field in __cuda_array_interface__.

    The idea is to delay the access to `Buffer.ptr` until the user
    actually accesses the data pointer.

    For instance, in many cases __cuda_array_interface__ is accessed
    only to determine whether an object is a CUDA object or not.
    """

    def __init__(self, buffer: Buffer) -> None:
        self._buf = buffer

    def __len__(self):
        return 2

    def __getitem__(self, i):
        if i == 0:
            return self._buf.ptr
        elif i == 1:
            return False
        raise IndexError("tuple index out of range")


class Buffer(Serializable):
    """
    A Buffer represents a device memory allocation.

    Parameters
    ----------
    data : Buffer, array_like, int
        An array-like object or integer representing a
        device or host pointer to pre-allocated memory.
    size : int, optional
        Size of memory allocation. Required if a pointer
        is passed for `data`.
    owner : object, optional
        Python object to which the lifetime of the memory
        allocation is tied. If provided, a reference to this
        object is kept in this Buffer.
    ptr_exposed : bool, optional
        Whether or not a raw pointer (integer or C pointer) has
        been exposed to the outside world. If this is the case,
        the buffer cannot be spilled.
    """

    _lock: RLock
    _ptr: Optional[int]  # Guarded by `_lock`
    _ptr_desc: dict  # Guarded by `_lock`
    _ptr_exposed: bool  # Guarded by `_lock`
    _size: int  # read-only
    _owner: object  # read-only
    _view_desc: Optional[dict]  # read-only
    _spill_manager: Optional[SpillManager]  # read-only
    _access_counter: AccessCounter
    _last_accessed: float

    def __init__(
        self,
        data: Any = None,
        size: int = None,
        owner: object = None,
        ptr_exposed: bool = True,
    ):
        from cudf._lib.column import AccessCounter
        from cudf.core.spill_manager import global_manager

        self._lock = RLock()
        self._access_counter = AccessCounter()
        self._ptr_exposed = ptr_exposed
        self._ptr_desc = {"type": "gpu"}
        self._last_accessed = time.monotonic()
        self._view_desc = (
            None  # TODO: make a view its own subclass `BufferView`
        )

        if isinstance(data, Buffer):
            self._size = data.size
            if ptr_exposed or owner:
                self._ptr = data.ptr  # Exposing `data`
                self._owner = owner or data._owner
            else:
                # Create a new buffer view that spans all of `data`
                if data._view_desc is None:
                    self._view_desc = {"base": data, "offset": 0}
                else:
                    self._view_desc = data._view_desc.copy()
                self._ptr = None
                self._owner = None
        elif isinstance(data, rmm.DeviceBuffer):
            self._ptr = data.ptr
            self._size = data.size
            self._owner = data
        elif hasattr(data, "__array_interface__") or hasattr(
            data, "__cuda_array_interface__"
        ):
            self._init_from_array_like(data, owner)
        elif isinstance(data, memoryview):
            if ptr_exposed:
                self._init_from_array_like(np.asarray(data), owner)
            else:
                # Create an already spilled Buffer
                self._ptr_desc = {"type": "cpu", "memoryview": data}
                self._ptr = None
                self._size = data.nbytes
                self._owner = None
        elif isinstance(data, int):
            if not isinstance(size, int):
                raise TypeError("size must be integer")
            self._ptr = data
            self._size = size
            self._owner = owner
        elif data is None:
            self._ptr = 0
            self._size = 0
            self._owner = None
        else:
            try:
                data = memoryview(data)
            except TypeError:
                raise TypeError("data must be Buffer, array-like or integer")
            self._init_from_array_like(np.asarray(data), owner)

        self._spill_manager = None
        if global_manager.enabled:
            self._spill_manager = global_manager.get()
            if self._view_desc is None and data is not None:
                base = None
                if self._ptr:
                    base = self._spill_manager.lookup_address_range(
                        self._ptr, self._size
                    )
                if base is not None:
                    base.ptr  # expose base buffer
                elif not self._ptr_exposed:
                    # `self` is a base buffer that hasn't been exposed
                    self._spill_manager.add(self)

    @classmethod
    def from_buffer(cls, buffer: Buffer, size: int = None, offset: int = 0):
        """
        Create a buffer from another buffer

        Parameters
        ----------
        buffer : Buffer
            The base buffer, which will also be set as the owner of
            the memory allocation.
        size : int, optional
            Size of the memory allocation (default: `buffer.size`).
        offset : int, optional
            Start offset relative to `buffer.ptr`.
        """

        ret = cls(ptr_exposed=False)
        ret._size = buffer.size if size is None else size
        ret._owner = buffer
        if ret._spill_manager is None or buffer.ptr_exposed:
            ret._ptr = buffer.ptr + offset
            return ret

        # Get base buffer
        if buffer._view_desc is None:
            base = buffer
            base_offset = 0
        else:
            base = buffer._view_desc["base"]
            base_offset = buffer._view_desc["offset"]

        ret._view_desc = {"base": base, "offset": base_offset + offset}
        return ret

    def __len__(self) -> int:
        return self._size

    @property
    def is_spilled(self) -> bool:
        if self._view_desc:
            return self._view_desc["base"].is_spilled
        return self._ptr_desc["type"] != "gpu"

    def move_inplace(self, target: str = "cpu") -> None:
        assert self._view_desc is None
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
                self._ptr = None
                self._owner = None
            elif (ptr_type, target) == ("cpu", "gpu"):
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

        Consider using `.restricted_ptr` instead.
        """
        if self._view_desc:
            return self._view_desc["base"].ptr + self._view_desc["offset"]
        if self._spill_manager is not None:
            self._spill_manager.spill_to_device_limit()
        with self._lock:
            self.move_inplace(target="gpu")
            self._ptr_exposed = True
            self._last_accessed = time.monotonic()
            assert self._ptr is not None
            return self._ptr

    @property
    def restricted_ptr(self):
        """Access the memory without exposing the buffer permanently"""
        raise NotImplementedError("TODO: implement")

    @property
    def ptr_exposed(self) -> bool:
        if self._view_desc:
            return self._view_desc["base"].ptr_exposed
        return self._ptr_exposed

    @property
    def spillable(self) -> bool:
        if self._view_desc:
            return self._view_desc["base"].spillable
        return not self._ptr_exposed and self._access_counter.use_count() == 1

    @property
    def size(self) -> int:
        return self._size

    @property
    def nbytes(self) -> int:
        return self._size

    @property
    def last_accessed(self) -> float:
        if self._view_desc:
            return self._view_desc["base"]._last_accessed
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

    def to_host_array(self):
        data = np.empty((self.size,), "u1")
        rmm._lib.device_buffer.copy_ptr_to_host(self.ptr, data)
        return data

    def _init_from_array_like(self, data, owner):

        if hasattr(data, "__cuda_array_interface__"):
            confirm_1d_contiguous(data.__cuda_array_interface__)
            ptr, size = _buffer_data_from_array_interface(
                data.__cuda_array_interface__
            )
            self._ptr = ptr
            self._size = size
            self._owner = owner or data
        elif hasattr(data, "__array_interface__"):
            confirm_1d_contiguous(data.__array_interface__)
            ptr, size = _buffer_data_from_array_interface(
                data.__array_interface__
            )
            dbuf = rmm.DeviceBuffer(ptr=ptr, size=size)
            self._init_from_array_like(dbuf, owner)
        else:
            raise TypeError(
                f"Cannot construct Buffer from {data.__class__.__name__}"
            )

    def serialize(self) -> Tuple[dict, list]:
        header = {}  # type: Dict[Any, Any]
        header["type-serialized"] = pickle.dumps(type(self))
        header["desc"] = {"shape": (self.size,)}
        header["desc"]["strides"] = (1,)
        header["frame_count"] = 1
        frames = [self]
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> Buffer:
        assert (
            header["frame_count"] == 1
        ), "Only expecting to deserialize Buffer with a single frame."
        buf = cls(frames[0], ptr_exposed=False)
        if header["desc"]["shape"] != buf.__cuda_array_interface__["shape"]:
            raise ValueError(
                f"Received a `Buffer` with the wrong size."
                f" Expected {header['desc']['shape']}, "
                f"but got {buf.__cuda_array_interface__['shape']}"
            )
        return buf

    def memoryview_read_only(self) -> memoryview:
        # Get base buffer
        if self._view_desc is None:
            base = self
            offset = 0
        else:
            base = self._view_desc["base"]
            offset = self._view_desc["offset"]

        with base._lock:
            if base.spillable:
                base.move_inplace(target="cpu")
                return base._ptr_desc["memoryview"][
                    offset : offset + self.size
                ]
            else:
                assert base._ptr_desc["type"] == "gpu"
                assert base._ptr is not None
                ret = memoryview(bytearray(self.size))
                rmm._lib.device_buffer.copy_ptr_to_host(
                    base._ptr + offset, ret
                )
                return ret

    @classmethod
    def empty(cls, size: int) -> Buffer:
        return Buffer(rmm.DeviceBuffer(size=size), ptr_exposed=False)

    def copy(self) -> Buffer:
        """
        Create a new Buffer containing a copy of the data contained
        in this Buffer.
        """
        from rmm._lib.device_buffer import copy_device_to_ptr

        out = Buffer.empty(size=self.size)
        copy_device_to_ptr(self.ptr, out.ptr, self.size)
        return out

    def __repr__(self) -> str:
        if self._view_desc is None:
            if self._ptr is None:
                ptr_info = str(self._ptr_desc)
            else:
                ptr_info = str(hex(self._ptr))
        else:
            ptr_info = str(self._view_desc)
        return (
            f"<cudf.core.buffer.Buffer size={format_bytes(self._size)} "
            f"spillable={self.spillable} ptr_exposed={self.ptr_exposed} "
            f"access_counter={self._access_counter.use_count()} "
            f"ptr={ptr_info} owner={repr(self._owner)} "
        )


def _buffer_data_from_array_interface(array_interface):
    ptr = array_interface["data"][0]
    if ptr is None:
        ptr = 0
    itemsize = cudf.dtype(array_interface["typestr"]).itemsize
    shape = (
        array_interface["shape"] if len(array_interface["shape"]) > 0 else (1,)
    )
    size = functools.reduce(operator.mul, shape)
    return ptr, size * itemsize


def confirm_1d_contiguous(array_interface):
    strides = array_interface["strides"]
    shape = array_interface["shape"]
    itemsize = cudf.dtype(array_interface["typestr"]).itemsize
    typestr = array_interface["typestr"]
    if typestr not in ("|i1", "|u1"):
        raise TypeError("Buffer data must be of uint8 type")
    if not get_c_contiguity(shape, strides, itemsize):
        raise ValueError("Buffer data must be 1D C-contiguous")


def get_c_contiguity(shape, strides, itemsize):
    """
    Determine if combination of array parameters represents a
    c-contiguous array.
    """
    ndim = len(shape)
    assert strides is None or ndim == len(strides)

    if ndim == 0 or strides is None or (ndim == 1 and strides[0] == itemsize):
        return True

    # any dimension zero, trivial case
    for dim in shape:
        if dim == 0:
            return True

    for this_dim, this_stride in zip(shape, strides):
        if this_stride != this_dim * itemsize:
            return False
    return True
