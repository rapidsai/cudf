# Copyright (c) 2022, NVIDIA CORPORATION.

from __future__ import annotations

from typing import Any, Union

from cudf.core.buffer.buffer import Buffer, cuda_array_interface_wrapper


def as_buffer(
    data: Union[int, Any],
    *,
    size: int = None,
    owner: object = None,
) -> Buffer:
    """Factory function to wrap `data` in a Buffer object.

    If `data` isn't a buffer already, a new buffer that points to the memory of
    `data` is created. If `data` represents host memory, it is copied to a new
    `rmm.DeviceBuffer` device allocation. Otherwise, the memory of `data` is
    **not** copied, instead the new buffer keeps a reference to `data` in order
    to retain its lifetime.

    If `data` is an integer, it is assumed to point to device memory.

    Raises ValueError if data isn't C-contiguous.

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

    Return
    ------
    Buffer
        A buffer instance that represents the device memory of `data`.
    """

    if isinstance(data, Buffer):
        return data

    # We handle the integer argument in the factory function by wrapping
    # the pointer in a `__cuda_array_interface__` exposing object so that
    # the Buffer (and its sub-classes) do not have to.
    if isinstance(data, int):
        if size is None:
            raise ValueError(
                "size must be specified when `data` is an integer"
            )
        data = cuda_array_interface_wrapper(ptr=data, size=size, owner=owner)
    elif size is not None or owner is not None:
        raise ValueError(
            "`size` and `owner` must be None when "
            "`data` is a buffer-like or array-like object"
        )

    if hasattr(data, "__cuda_array_interface__"):
        return Buffer._from_device_memory(data)
    return Buffer._from_host_memory(data)
