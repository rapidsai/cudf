# Copyright (c) 2022, NVIDIA CORPORATION.

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, DefaultDict, Tuple, Type, TypeVar
from weakref import WeakSet

import rmm

from cudf.core.buffer.buffer import Buffer, cuda_array_interface_wrapper

T = TypeVar("T", bound="CopyOnWriteBuffer")


class CopyOnWriteBuffer(Buffer):
    """A Buffer represents device memory.

    Use the factory function `as_buffer` to create a Buffer instance.
    """

    # This dict keeps track of all instances that have the same `ptr`
    # and `size` attributes.  Each key of the dict is a `(ptr, size)`
    # tuple and the corresponding value is a set of weak references to
    # instances with that `ptr` and `size`.
    _instances: DefaultDict[Tuple, WeakSet] = defaultdict(WeakSet)

    # TODO: This is synonymous to SpillableBuffer._exposed attribute
    # and has to be merged.
    _zero_copied: bool

    def _finalize_init(self):
        # the last step in initializing a `CopyOnWriteBuffer`
        # is to track it in `_instances`:
        key = (self.ptr, self.size)
        self.__class__._instances[key].add(self)
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

    def _shallow_copied(self):
        """
        Return `True` if shallow copies of `self` exist.
        """
        return len(self.__class__._instances[(self.ptr, self.size)]) > 1

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
            owner_copy: rmm.DeviceBuffer = copy.copy(self._owner)
            return self._from_device_memory(
                cuda_array_interface_wrapper(
                    ptr=owner_copy.ptr,
                    size=owner_copy.size,
                    owner=owner_copy,
                )
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
        self._detach_refs(zero_copied=True)

        result = self._cuda_array_interface_readonly
        result["data"] = (self.ptr, False)
        return result

    def _detach_refs(self, zero_copied=False):
        """
        Detaches a Buffer from it's weak-references by making
        a true deep-copy.
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
