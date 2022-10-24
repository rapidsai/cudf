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
        """"""

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
    def __init__(self, ptr, size) -> None:
        self.ptr = ptr
        self.size = size


def custom_weakref_callback(ref):
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
        self._temp_ref = None
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
        return self._zero_copied

    def _update_ref(self):
        if (self._ptr, self._size) not in Buffer._refs:
            Buffer._refs[(self._ptr, self._size)] = BufferWeakref(
                self._ptr, self._size
            )
        self._temp_ref = Buffer._refs[(self._ptr, self._size)]

    def get_ref(self):
        if self._temp_ref is None:
            self._update_ref()
        return self._temp_ref

    def has_a_weakref(self):
        weakref_count = weakref.getweakrefcount(self.get_ref())

        if weakref_count == 1:
            return (
                not weakref.getweakrefs(self.get_ref())[0]()
                is not self.get_ref()
            )
        else:
            return weakref_count > 0

    def get_weakref(self):
        return weakref.ref(self.get_ref(), custom_weakref_callback)

    def copy(self, deep: bool = True):
        if deep:
            if (
                cudf.get_option("copy_on_write")
                and not self._is_cai_zero_copied()
            ):
                copied_buf = Buffer.__new__(Buffer)
                copied_buf._ptr = self._ptr
                copied_buf._size = self._size
                copied_buf._owner = self._owner
                copied_buf._temp_ref = None
                copied_buf._weak_ref = None
                copied_buf._zero_copied = False

                if self._weak_ref is None:
                    self._weak_ref = copied_buf.get_weakref()
                    copied_buf._weak_ref = self.get_weakref()
                else:
                    if self.has_a_weakref():
                        copied_buf._weak_ref = self._weak_ref
                        self._weak_ref = copied_buf.get_weakref()
                    else:
                        self._weak_ref = copied_buf.get_weakref()
                        copied_buf._weak_ref = self.get_weakref()
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
        return {
            "data": (self.ptr, False),
            "shape": (self.size,),
            "strides": None,
            "typestr": "|u1",
            "version": 0,
        }

    @property
    def __cuda_array_interface__(self) -> dict:
        self._detach_refs()
        self._zero_copied = True
        return self._cai

    def _detach_refs(self):
        if not self._zero_copied and self.has_a_weakref():
            # make a deep copy of existing DeviceBuffer
            # and replace pointer to it.
            current_buf = rmm.DeviceBuffer(ptr=self.ptr, size=self.size)
            new_buf = current_buf.copy()
            self._ptr = new_buf.ptr
            self._size = new_buf.size
            self._owner = new_buf

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
