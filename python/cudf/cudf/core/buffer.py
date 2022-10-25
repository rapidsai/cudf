# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from __future__ import annotations

import copy
import math
import pickle
import weakref
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np

import rmm

import cudf
from cudf.core.abc import Serializable
from cudf.utils.string import format_bytes

# Frame type for serialization and deserialization of `DeviceBufferLike`
Frame = Union[memoryview, "DeviceBufferLike"]


@runtime_checkable
class DeviceBufferLike(Protocol):
    def __getitem__(self, key: slice) -> DeviceBufferLike:
        """Create a new view of the buffer."""

    @property
    def size(self) -> int:
        """Size of the buffer in bytes."""

    @property
    def nbytes(self) -> int:
        """Size of the buffer in bytes."""

    @property
    def ptr(self) -> int:
        """Device pointer to the start of the buffer."""

    @property
    def owner(self) -> Any:
        """Object owning the memory of the buffer."""

    @property
    def __cuda_array_interface__(self) -> Mapping:
        """Implementation of the CUDA Array Interface."""

    @property
    def _cai(self) -> Mapping:
        """
        Internal Implementation for the CUDA Array Interface without
        triggering a deepcopy.
        """

    def copy(self, deep: bool = True) -> DeviceBufferLike:
        """Make a copy of Buffer."""

    def memoryview(self) -> memoryview:
        """Read-only access to the buffer through host memory."""

    def serialize(self) -> Tuple[dict, List[Frame]]:
        """Serialize the buffer into header and frames.

        The frames can be a mixture of memoryview and device-buffer-like
        objects.

        Returns
        -------
        Tuple[Dict, List]
            The first element of the returned tuple is a dict containing any
            serializable metadata required to reconstruct the object. The
            second element is a list containing the device buffers and
            memoryviews of the object.
        """

    @classmethod
    def deserialize(
        cls, header: dict, frames: List[Frame]
    ) -> DeviceBufferLike:
        """Generate an buffer from a serialized representation.

        Parameters
        ----------
        header : dict
            The metadata required to reconstruct the object.
        frames : list
            The device-buffer-like and memoryview buffers that the object
            should contain.

        Returns
        -------
        DeviceBufferLike
            A new object that implements DeviceBufferLike.
        """


def as_device_buffer_like(obj: Any) -> DeviceBufferLike:
    """
    Factory function to wrap `obj` in a DeviceBufferLike object.

    If `obj` isn't device-buffer-like already, a new buffer that implements
    DeviceBufferLike and points to the memory of `obj` is created. If `obj`
    represents host memory, it is copied to a new `rmm.DeviceBuffer` device
    allocation. Otherwise, the data of `obj` is **not** copied, instead the
    new buffer keeps a reference to `obj` in order to retain the lifetime
    of `obj`.

    Raises ValueError if the data of `obj` isn't C-contiguous.

    Parameters
    ----------
    obj : buffer-like or array-like
        An object that exposes either device or host memory through
        `__array_interface__`, `__cuda_array_interface__`, or the
        buffer protocol. If `obj` represents host memory, data will
        be copied.

    Return
    ------
    DeviceBufferLike
        A device-buffer-like instance that represents the device memory
        of `obj`.
    """

    if isinstance(obj, DeviceBufferLike):
        return obj
    return Buffer(obj)


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
    """
    A Buffer represents device memory.

    Usually Buffers will be created using `as_device_buffer_like(obj)`,
    which will make sure that `obj` is device-buffer-like and not a `Buffer`
    necessarily.

    Parameters
    ----------
    data : int or buffer-like or array-like
        An integer representing a pointer to device memory or a buffer-like
        or array-like object. When not an integer, `size` and `owner` must
        be None.
    size : int, optional
        Size of device memory in bytes. Must be specified if `data` is an
        integer.
    owner : object, optional
        Python object to which the lifetime of the memory allocation is tied.
        A reference to this object is kept in the returned Buffer.
    """

    _ptr: int
    _size: int
    _owner: object
    _refs: dict = {}

    def __init__(
        self, data: Union[int, Any], *, size: int = None, owner: object = None
    ):
        self._weak_ref = None
        self._proxy_ref = None
        self._zero_copied = False

        if isinstance(data, int):
            if size is None:
                raise ValueError(
                    "size must be specified when `data` is an integer"
                )
            if size < 0:
                raise ValueError("size cannot be negative")
            self._ptr = data
            self._size = size
            self._owner = owner
            self._update_ref()
        else:
            if size is not None or owner is not None:
                raise ValueError(
                    "`size` and `owner` must be None when "
                    "`data` is a buffer-like object"
                )

            # `data` is a buffer-like object
            buf: Any = data
            if isinstance(buf, (Buffer, rmm.DeviceBuffer)):
                self._ptr = buf.ptr
                self._size = buf.size
                self._owner = buf
                self._update_ref()
                return
            iface = getattr(buf, "__cuda_array_interface__", None)
            if iface:
                ptr, size = get_ptr_and_size(iface)
                self._ptr = ptr
                self._size = size
                self._owner = buf
                self._update_ref()
                return
            ptr, size = get_ptr_and_size(np.asarray(buf).__array_interface__)
            buf = rmm.DeviceBuffer(ptr=ptr, size=size)
            self._ptr = buf.ptr
            self._size = buf.size
            self._owner = buf
            self._update_ref()

    def __getitem__(self, key: slice) -> Buffer:
        if not isinstance(key, slice):
            raise ValueError("index must be an slice")
        start, stop, step = key.indices(self.size)
        if step != 1:
            raise ValueError("slice must be contiguous")
        return self.__class__(
            data=self.ptr + start, size=stop - start, owner=self.owner
        )

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

    def get_weakref(self):
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
                    copied_buf._weak_ref = self.get_weakref()

                self._weak_ref = copied_buf.get_weakref()

                return copied_buf
            else:
                owner_copy = copy.copy(self._owner)
                return Buffer(data=None, size=None, owner=owner_copy)
        else:
            shallow_copy = Buffer.__new__(Buffer)
            shallow_copy._ptr = self._ptr
            shallow_copy._size = self._size
            shallow_copy._owner = self._owner
            return shallow_copy

    @property
    def size(self) -> int:
        return self._size

    @property
    def nbytes(self) -> int:
        return self._size

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def owner(self) -> Any:
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
        host_buf = bytearray(self.size)
        rmm._lib.device_buffer.copy_ptr_to_host(self.ptr, host_buf)
        return memoryview(host_buf).toreadonly()

    def serialize(self) -> Tuple[dict, list]:
        header = {}  # type: Dict[Any, Any]
        header["type-serialized"] = pickle.dumps(type(self))
        header["constructor-kwargs"] = {}
        header["desc"] = self._cai.copy()
        header["desc"]["strides"] = (1,)
        header["frame_count"] = 1
        frames = [self]
        return header, frames

    @classmethod
    def deserialize(cls, header: dict, frames: list) -> Buffer:
        assert (
            header["frame_count"] == 1
        ), "Only expecting to deserialize Buffer with a single frame."
        buf = cls(frames[0], **header["constructor-kwargs"])

        if header["desc"]["shape"] != buf._cai["shape"]:
            raise ValueError(
                f"Received a `Buffer` with the wrong size."
                f" Expected {header['desc']['shape']}, "
                f"but got {buf._cai['shape']}"
            )

        return buf

    def __repr__(self) -> str:
        return (
            f"<cudf.core.buffer.Buffer size={format_bytes(self._size)} "
            f"ptr={hex(self._ptr)} owner={repr(self._owner)} "
        )


def is_c_contiguous(
    shape: Sequence[int], strides: Sequence[int], itemsize: int
) -> bool:
    """
    Determine if shape and strides are C-contiguous

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
    """
    Retrieve the pointer and size from an array interface.

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
