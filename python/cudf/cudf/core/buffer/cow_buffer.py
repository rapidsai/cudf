# Copyright (c) 2022, NVIDIA CORPORATION.

from __future__ import annotations

from typing import Any, Type, TypeVar
from weakref import WeakKeyDictionary, WeakSet

import rmm

from cudf.core.buffer.buffer import Buffer

T = TypeVar("T", bound="CopyOnWriteBuffer")


class _PtrSize:
    __slots__ = ("t", "__weakref__")

    def __init__(self, ptr, size):
        self.t = (ptr, size)

    def __hash__(self):
        return hash(self.t)

    def __eq__(self, o):
        return type(o) is type(self) and o.t == self.t

    def __repr__(self):
        return f"PtrSize{self.t!r}"

    def __str__(self):
        return f"PtrSize{self.t}"


class CopyOnWriteBuffer(Buffer):
    """A Buffer represents device memory.

    Use the factory function `as_buffer` to create a Buffer instance.
    """

    # This dict keeps track of all instances that have the same `ptr`
    # and `size` attributes.  Each key of the dict is a `(ptr, size)`
    # tuple and the corresponding value is a set of weak references to
    # instances with that `ptr` and `size`.
    _instances: WeakKeyDictionary = WeakKeyDictionary()

    # TODO: This is synonymous to SpillableBuffer._exposed attribute
    # and has to be merged.
    _zero_copied: bool
    _ptrsize: _PtrSize

    def _finalize_init(self):
        # the last step in initializing a `CopyOnWriteBuffer`
        # is to track it in `_instances`:
        self._ptrsize = _PtrSize(self.ptr, self.size)
        self.__class__._instances.setdefault(self._ptrsize, WeakSet()).add(
            self
        )
        self._zero_copied = False

    @classmethod
    def _from_device_memory(cls: Type[T], data: Any) -> T:
        """Create a Buffer from an object exposing `__cuda_array_interface__`.

        No data is being copied.

        Parameters
        ----------
        data : device-buffer-like
            An object implementing the CUDA Array Interface.

        Returns
        -------
        Buffer
            Buffer representing the same device memory as `data`
        """

        # Bypass `__init__` and initialize attributes manually
        ret = super()._from_device_memory(data)
        ret._finalize_init()
        return ret

    @classmethod
    def _from_host_memory(cls: Type[T], data: Any) -> T:
        ret = super()._from_host_memory(data)
        ret._finalize_init()
        return ret

    @property
    def _is_shared(self):
        """
        Return `True` if `self`'s memory is shared with other columns.
        """
        return len(self.__class__._instances[self._ptrsize]) > 1

    def copy(self, deep: bool = True):
        """
        Return a copy of Buffer.

        Parameters
        ----------
        deep : bool, default True
            If True, returns a deep-copy of the underlying Buffer data.
            If False, returns a shallow-copy of the Buffer pointing to
            the same underlying data.

        Returns
        -------
        Buffer
        """
        if not deep:
            copied_buf = CopyOnWriteBuffer.__new__(CopyOnWriteBuffer)
            copied_buf._ptr = self._ptr
            copied_buf._size = self._size
            copied_buf._owner = self._owner
            copied_buf._finalize_init()
            return copied_buf
        else:
            return self._from_device_memory(
                rmm.DeviceBuffer(ptr=self.ptr, size=self.size)
            )

    @property
    def _cuda_array_interface_readonly(self) -> dict:
        """
        Internal Implementation for the CUDA Array Interface without
        triggering a deepcopy.
        """
        return {
            "data": (self.ptr, True),
            "shape": (self.size,),
            "strides": None,
            "typestr": "|u1",
            "version": 0,
        }

    @property
    def __cuda_array_interface__(self) -> dict:
        # Detach if there are any weak-references.

        # Mark the Buffer as ``zero_copied=True``,
        # which will prevent any copy-on-write
        # mechanism post this operation.
        # This is done because we don't have any
        # control over knowing if a third-party library
        # has modified the data this Buffer is
        # pointing to.
        self._unlink_shared_buffers(zero_copied=True)

        result = self._cuda_array_interface_readonly
        result["data"] = (self.ptr, False)
        return result

    def _unlink_shared_buffers(self, zero_copied=False):
        """
        Unlinks a Buffer if it is shared with other buffers(i.e.,
        weak references exist) by making a true deep-copy.
        """
        if not self._zero_copied and self._shallow_copied():
            # make a deep copy of existing DeviceBuffer
            # and replace pointer to it.
            current_buf = rmm.DeviceBuffer(ptr=self.ptr, size=self.size)
            new_buf = current_buf.copy()
            self._ptr = new_buf.ptr
            self._size = new_buf.size
            self._owner = new_buf
            self._finalize_init()
        self._zero_copied = zero_copied
