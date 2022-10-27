# Copyright (c) 2022, NVIDIA CORPORATION.

from __future__ import annotations

from typing import Any

from cudf.core.buffer.buffer import Buffer


def as_buffer(obj: Any) -> Buffer:
    """
    Factory function to wrap `obj` in a Buffer object.

    If `obj` isn't buffer already, a new buffer that points to the memory of
    `obj` is created. If `obj` represents host memory, it is copied to a new
    `rmm.DeviceBuffer` device allocation. Otherwise, the data of `obj` is
    **not** copied, instead the new buffer keeps a reference to `obj` in order
    to retain the lifetime of `obj`.

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
    Buffer
        A buffer instance that represents the device memory
        of `obj`.
    """

    if isinstance(obj, Buffer):
        return obj
    return Buffer(obj)
