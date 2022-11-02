# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from __future__ import annotations

import copy
import math
import pickle
import weakref
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence, Tuple, Type, TypeVar

import numpy

import rmm

import cudf
from cudf.core.abc import Serializable
from cudf.utils.string import format_bytes

T = TypeVar("T", bound="Buffer")


def cuda_array_interface_wrapper(
    ptr: int,
    size: int,
    owner: object = None,
    readonly=False,
    typestr="|u1",
    version=0,
):
    """Wrap device pointer in an object that exposes `__cuda_array_interface__`

    See <https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html>

    Parameters
    ----------
    ptr : int
        An integer representing a pointer to device memory.
    size : int, optional
        Size of device memory in bytes.
    owner : object, optional
        Python object to which the lifetime of the memory allocation is tied.
        A reference to this object is kept in the returned wrapper object.
    readonly: bool, optional
        Mark the interface read-only.
    typestr: str, optional
        The type string of the interface. By default this is "|u1", which
        means "an unsigned integer with a not relevant byteorder". See:
        <https://numpy.org/doc/stable/reference/arrays.interface.html>
    version : bool, optional
        The version of the interface.

    Return
    ------
    SimpleNamespace
        An object that exposes `__cuda_array_interface__` and keeps a reference
        to `owner`.
    """

    if size < 0:
        raise ValueError("size cannot be negative")

    return SimpleNamespace(
        __cuda_array_interface__={
            "data": (ptr, readonly),
            "shape": (size,),
            "strides": None,
            "typestr": typestr,
            "version": version,
        },
        owner=owner,
    )


class BufferWeakref(object):
    """
    A proxy class to be used by ``Buffer`` for generating weakreferences.
    """

    def __init__(self, ptr, size) -> None:
        self.ptr = ptr
        self.size = size


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


class Buffer(Serializable):
    """A Buffer represents device memory.

    Use the factory function `as_buffer` to create a Buffer instance.
    """

    _ptr: int
    _size: int
    _owner: object
    _weak_ref: object
    _proxy_ref: None | BufferWeakref
    _zero_copied: bool
    _refs: dict = {}

    def __init__(self):
        raise ValueError(
            f"do not create a {self.__class__} directly, please "
            "use the factory function `cudf.core.buffer.as_buffer`"
        )

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
        ret = cls.__new__(cls)
        ret._owner = data
        ret._weak_ref = None
        ret._proxy_ref = None
        ret._zero_copied = False
        if isinstance(data, rmm.DeviceBuffer):  # Common case shortcut
            ret._ptr = data.ptr
            ret._size = data.size
        else:
            ret._ptr, ret._size = get_ptr_and_size(
                data.__cuda_array_interface__
            )
        if ret.size < 0:
            raise ValueError("size cannot be negative")
        ret._update_ref()
        return ret

    @classmethod
    def _from_host_memory(cls: Type[T], data: Any) -> T:
        """Create a Buffer from a buffer or array like object

        Data must implement `__array_interface__`, the buffer protocol, and/or
        be convertible to a buffer object using `numpy.array()`

        The host memory is copied to a new device allocation.

        Raises ValueError if array isn't C-contiguous.

        Parameters
        ----------
        data : Any
            An object that represens host memory.

        Returns
        -------
        Buffer
            Buffer representing a copy of `data`.
        """

        # Convert to numpy array, this will not copy data in most cases.
        ary = numpy.array(data, copy=False, subok=True)
        # Extract pointer and size
        ptr, size = get_ptr_and_size(ary.__array_interface__)
        # Copy to device memory
        buf = rmm.DeviceBuffer(ptr=ptr, size=size)
        # Create from device memory
        return cls._from_device_memory(buf)

    def _getitem(self, offset: int, size: int) -> Buffer:
        """
        Sub-classes can overwrite this to implement __getitem__
        without having to handle non-slice inputs.
        """
        return self._from_device_memory(
            cuda_array_interface_wrapper(
                ptr=self.ptr + offset, size=size, owner=self.owner
            )
        )

    def __getitem__(self, key: slice) -> Buffer:
        """Create a new slice of the buffer."""
        if not isinstance(key, slice):
            raise TypeError(
                "Argument 'key' has incorrect type "
                f"(expected slice, got {key.__class__.__name__})"
            )
        start, stop, step = key.indices(self.size)
        if step != 1:
            raise ValueError("slice must be C-contiguous")
        return self._getitem(offset=start, size=stop - start)

    def _is_cai_zero_copied(self):
        """
        Returns a flag, that indicates if the Buffer has been zero-copied.
        """
        return self._zero_copied

    def _update_ref(self):
        """
        Generate the new proxy reference.
        """
        if (self._ptr, self._size) not in Buffer._refs:
            Buffer._refs[(self._ptr, self._size)] = BufferWeakref(
                self._ptr, self._size
            )
        self._proxy_ref = Buffer._refs[(self._ptr, self._size)]

    def get_ref(self):
        """
        Returns the proxy reference.
        """
        if self._proxy_ref is None:
            self._update_ref()
        return self._proxy_ref

    def has_a_weakref(self):
        """
        Checks if the Buffer has a weak-reference.
        """
        weakref_count = weakref.getweakrefcount(self.get_ref())

        if weakref_count == 1:
            # When the weakref_count is 1, it could be a possibility
            # that a copied Buffer got destroyed and hence this
            # method should return False in that case as there is only
            # one Buffer pointing to the device memory.
            return (
                weakref.getweakrefs(self.get_ref())[0]() is not self.get_ref()
            )
        else:
            return weakref_count > 0

    def _get_weakref(self):
        """
        Returns a weak-reference for the Buffer.
        """
        return weakref.ref(self.get_ref(), custom_weakref_callback)

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
        if deep:
            if (
                cudf.get_option("copy_on_write")
                and not self._is_cai_zero_copied()
            ):
                copied_buf = Buffer.__new__(Buffer)
                copied_buf._ptr = self._ptr
                copied_buf._size = self._size
                copied_buf._owner = self._owner
                copied_buf._proxy_ref = None
                copied_buf._weak_ref = None
                copied_buf._zero_copied = False

                if self.has_a_weakref():
                    # If `self` has weak-references
                    # we will then have to keep that
                    # weak-reference alive, hence
                    # pass it onto `copied_buf`
                    copied_buf._weak_ref = self._weak_ref
                else:
                    # If `self` has no weak-references,
                    # we will have to generate a new weak-reference
                    # and assign it to `copied_buf`
                    copied_buf._weak_ref = self._get_weakref()

                self._weak_ref = copied_buf._get_weakref()

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
        else:
            shallow_copy = Buffer.__new__(Buffer)
            shallow_copy._ptr = self._ptr
            shallow_copy._size = self._size
            shallow_copy._owner = self._owner
            return shallow_copy

    @property
    def size(self) -> int:
        """Size of the buffer in bytes."""
        return self._size

    @property
    def nbytes(self) -> int:
        """Size of the buffer in bytes."""
        return self._size

    @property
    def ptr(self) -> int:
        """Device pointer to the start of the buffer."""
        return self._ptr

    @property
    def owner(self) -> Any:
        """Object owning the memory of the buffer."""
        return self._owner

    @property
    def _cai(self) -> dict:
        """
        Internal Implementation for the CUDA Array Interface without
        triggering a deepcopy.
        """
        return {
            "data": (self.ptr, False),
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

        return self._cai

    def _detach_refs(self, zero_copied=False):
        """
        Detaches a Buffer from it's weak-references by making
        a true deep-copy.
        """
        if not self._zero_copied and self.has_a_weakref():
            # make a deep copy of existing DeviceBuffer
            # and replace pointer to it.
            current_buf = rmm.DeviceBuffer(ptr=self.ptr, size=self.size)
            new_buf = current_buf.copy()
            self._ptr = new_buf.ptr
            self._size = new_buf.size
            self._owner = new_buf
        self._zero_copied = zero_copied

    def memoryview(self) -> memoryview:
        """Read-only access to the buffer through host memory."""
        host_buf = bytearray(self.size)
        rmm._lib.device_buffer.copy_ptr_to_host(self.ptr, host_buf)
        return memoryview(host_buf).toreadonly()

    def serialize(self) -> Tuple[dict, list]:
        """Serialize the buffer into header and frames.

        The frames can be a mixture of memoryview and Buffer objects.

        Returns
        -------
        Tuple[dict, List]
            The first element of the returned tuple is a dict containing any
            serializable metadata required to reconstruct the object. The
            second element is a list containing Buffers and memoryviews.
        """
        header: Dict[str, Any] = {}
        header["type-serialized"] = pickle.dumps(type(self))
        header["frame_count"] = 1
        frames = [self]
        return header, frames

    @classmethod
    def deserialize(cls: Type[T], header: dict, frames: list) -> T:
        """Create an Buffer from a serialized representation.

        Parameters
        ----------
        header : dict
            The metadata required to reconstruct the object.
        frames : list
            The Buffer and memoryview that makes up the Buffer.

        Returns
        -------
        Buffer
            The deserialized Buffer.
        """
        if header["frame_count"] != 1:
            raise ValueError("Deserializing a Buffer expect a single frame")
        frame = frames[0]
        if isinstance(frame, cls):
            return frame  # The frame is already deserialized

        if hasattr(frame, "__cuda_array_interface__"):
            return cls._from_device_memory(frame)
        return cls._from_host_memory(frame)

    def __repr__(self) -> str:
        klass = self.__class__
        name = f"{klass.__module__}.{klass.__qualname__}"
        return (
            f"<{name} size={format_bytes(self._size)} "
            f"ptr={hex(self._ptr)} owner={repr(self._owner)}>"
        )


def is_c_contiguous(
    shape: Sequence[int], strides: Sequence[int], itemsize: int
) -> bool:
    """Determine if shape and strides are C-contiguous

    Parameters
    ----------
    shape : Sequence[int]
        Number of elements in each dimension.
    strides : Sequence[int]
        The stride of each dimension in bytes.
    itemsize : int
        Size of an element in bytes.

    Return
    ------
    bool
        The boolean answer.
    """

    if any(dim == 0 for dim in shape):
        return True
    cumulative_stride = itemsize
    for dim, stride in zip(reversed(shape), reversed(strides)):
        if dim > 1 and stride != cumulative_stride:
            return False
        cumulative_stride *= dim
    return True


def get_ptr_and_size(array_interface: Mapping) -> Tuple[int, int]:
    """Retrieve the pointer and size from an array interface.

    Raises ValueError if array isn't C-contiguous.

    Parameters
    ----------
    array_interface : Mapping
        The array interface metadata.

    Return
    ------
    pointer : int
        The pointer to device or host memory
    size : int
        The size in bytes
    """

    shape = array_interface["shape"] or (1,)
    strides = array_interface["strides"]
    itemsize = cudf.dtype(array_interface["typestr"]).itemsize
    if strides is None or is_c_contiguous(shape, strides, itemsize):
        nelem = math.prod(shape)
        ptr = array_interface["data"][0] or 0
        return ptr, nelem * itemsize
    raise ValueError("Buffer data must be C-contiguous")
