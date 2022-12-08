# Copyright (c) 2022, NVIDIA CORPORATION.

from __future__ import annotations

import copy
import weakref
from typing import Any, Dict, Tuple, Type, TypeVar

import rmm

import cudf
from cudf.core.buffer.buffer import Buffer, cuda_array_interface_wrapper

T = TypeVar("T", bound="CopyOnWriteBuffer")


class _InstanceCountableMeta(type):
    """
    Metaclass that allows counting the number of instances that are
    constructed with the same `ptr` and `size`.
    """

    __weakrefs: Dict[Tuple, Any] = {}

    def __call__(cls, ptr, size):
        obj = super().__call__(ptr, size)
        key = (ptr, size)
        if key not in cls.__weakrefs:
            cls.__weakrefs[key] = weakref.WeakSet()
        cls.__weakrefs[key].add(obj)
        return obj

    def _instance_count(cls, ptr, size):
        return len(cls.__weakrefs[(ptr, size)])


class _BufferInstanceProxy(metaclass=_InstanceCountableMeta):
    """
    A proxy class used to count the number of instances of a
    `CopyOnWriteBuffer` constructed with the same `ptr` and `size`.
    """

    __slots__ = ("ptr", "size", "__weakref__")

    def __init__(self, ptr, size) -> None:
        self.ptr = ptr
        self.size = size

    def instance_count(self):
        return self.__class__._instance_count(self.ptr, self.size)


def custom_weakref_callback(ref):
    """
    A callback for ``weakref.ref`` API to generate unique
    weakref instances that can be counted correctly.

    Example below shows why this is necessary:

    In [1]: import cudf
    In [2]: import weakref

    Let's create an object ``x`` that we are going to weakref:

    In [3]: x = cudf.core.buffer.BufferWeakref(1, 2)

    Now generate three weak-references of it:

    In [4]: a = weakref.ref(x)
    In [5]: b = weakref.ref(x)
    In [6]: c = weakref.ref(x)

    ``weakref.ref`` actually returns the same singleton object:

    In [7]: a
    Out[7]: <weakref at 0x7f5bea052400; to 'BufferWeakref' at 0x7f5c99ecd850>
    In [8]: b
    Out[8]: <weakref at 0x7f5bea052400; to 'BufferWeakref' at 0x7f5c99ecd850>
    In [9]: c
    Out[9]: <weakref at 0x7f5bea052400; to 'BufferWeakref' at 0x7f5c99ecd850>

    In [10]: a is b
    Out[10]: True
    In [11]: b is c
    Out[11]: True

    This will be problematic as we cannot determine what is the count
    of weak-references:

    In [12]: weakref.getweakrefcount(x)
    Out[12]: 1

    Notice, though we want ``weakref.getweakrefcount`` to return ``3``, it
    returns ``1``. So we need to work-around this by using an empty/no-op
    callback:

    In [13]: def custom_weakref_callback(ref):
        ...:     pass
        ...:


    In [14]: d = weakref.ref(x, custom_weakref_callback)
    In [15]: e = weakref.ref(x, custom_weakref_callback)
    In [16]: f = weakref.ref(x, custom_weakref_callback)

    Now there is an each unique weak-reference created:

    In [17]: d
    Out[17]: <weakref at 0x7f5beb03e360; to 'BufferWeakref' at 0x7f5c99ecd850>
    In [18]: e
    Out[18]: <weakref at 0x7f5bd15e3810; to 'BufferWeakref' at 0x7f5c99ecd850>
    In [19]: f
    Out[19]: <weakref at 0x7f5bd15f1f40; to 'BufferWeakref' at 0x7f5c99ecd850>

    Now calling ``weakref.getweakrefcount`` will result in ``4``, which is correct:

    In [20]: weakref.getweakrefcount(x)
    Out[20]: 4

    In [21]: d is not e
    Out[21]: True

    In [22]: d is not f
    Out[22]: True

    In [23]: e is not f
    Out[23]: True
    """  # noqa: E501
    pass


class CopyOnWriteBuffer(Buffer):
    """A Buffer represents device memory.

    Use the factory function `as_buffer` to create a Buffer instance.
    """

    _proxy_ref: _BufferInstanceProxy
    # TODO: This is synonymous to SpillableBuffer._exposed attribute
    # and has to be merged.
    _zero_copied: bool

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
        ret._proxy_ref = _BufferInstanceProxy(ret._ptr, ret._size)
        ret._zero_copied = False
        return ret

    def _shallow_copied(self):
        return self._proxy_ref.instance_count() > 1

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
            if cudf.get_option("copy_on_write") and not self._zero_copied:
                copied_buf = CopyOnWriteBuffer.__new__(CopyOnWriteBuffer)
                copied_buf._ptr = self._ptr
                copied_buf._size = self._size
                copied_buf._owner = self._owner
                copied_buf._zero_copied = False
                # make the `_proxy_ref` of the copy a new instance:
                copied_buf._proxy_ref = _BufferInstanceProxy(
                    self._ptr, self._size
                )
                return copied_buf
            else:
                shallow_copy = CopyOnWriteBuffer.__new__(CopyOnWriteBuffer)
                shallow_copy._ptr = self._ptr
                shallow_copy._size = self._size
                shallow_copy._owner = self._owner
                # when shallow copying, don't make a new instance:
                shallow_copy._proxy_ref = self._proxy_ref
                return shallow_copy
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
            self._proxy_ref = _BufferInstanceProxy(self._ptr, self._size)
            self._owner = new_buf
        self._zero_copied = zero_copied
