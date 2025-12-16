# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Span protocol for objects providing pointer and size access."""

from typing import Any, Protocol, TypeGuard, runtime_checkable


@runtime_checkable
class Span(Protocol):
    """
    Protocol for objects that provide direct pointer and size access.

    This protocol is designed for zero-overhead access in Cython code,
    where ptr and size can be accessed as C attributes without Python overhead.
    Objects implementing this protocol must provide:

    - ptr: An integer pointer (uintptr_t) to device or host memory
    - size: Size of the memory region in bytes (uint64_t)

    The Span protocol enables pylibcudf.Column to accept arbitrary buffer types
    beyond gpumemoryview, including cudf Buffers and custom wrapper objects.

    Notes
    -----
    This protocol uses runtime_checkable for isinstance checks, but
    performance-critical code should use the is_span() TypeGuard instead,
    which uses fast hasattr checks.

    Examples
    --------
    >>> from pylibcudf.span import Span, is_span
    >>> import rmm
    >>> from pylibcudf.gpumemoryview import gpumemoryview
    >>>
    >>> # gpumemoryview implements Span
    >>> buf = rmm.DeviceBuffer(size=1024)
    >>> gmv = gpumemoryview(buf)
    >>> assert is_span(gmv)
    >>> assert gmv.ptr != 0
    >>> assert gmv.size == 1024
    """

    @property
    def ptr(self) -> int:
        """Device or host pointer as an integer (uintptr_t)."""
        ...

    @property
    def size(self) -> int:
        """Size of the memory region in bytes."""
        ...


def is_span(obj: Any) -> TypeGuard[Span]:
    """
    Runtime check for Span protocol compliance using TypeGuard.

    This function is optimized for performance using hasattr checks
    instead of isinstance(obj, Span), which is slow at runtime due
    to Protocol's structural subtyping overhead.

    Parameters
    ----------
    obj : Any
        Object to check for Span protocol compliance

    Returns
    -------
    TypeGuard[Span]
        True if obj has 'ptr' and 'size' attributes

    Examples
    --------
    >>> class MockSpan:
    ...     ptr = 0x1234
    ...     size = 1024
    >>> buffer = MockSpan()
    >>> if is_span(buffer):
    ...     # Type checker knows buffer is a Span here
    ...     print(f"ptr: {buffer.ptr}, size: {buffer.size}")
    ptr: 4660, size: 1024
    """
    return hasattr(obj, "ptr") and hasattr(obj, "size")
