# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for column operations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from typing import Any


def _flatten_and_access(
    objs: tuple[Any, ...], stack: ExitStack, kwargs: dict[str, Any]
) -> Iterator[Any]:
    """Recursively flatten varargs and enter access contexts for columns.

    Yields all objects in order. For objects with .access() method, enters
    their access context and yields the accessed object. For other objects,
    yields them unchanged.
    """
    for obj in objs:
        # Check if object has .access() method (duck typing)
        if (access := getattr(obj, "access", None)) is not None and callable(
            access
        ):
            # Enter access context and yield the accessed object
            accessed_obj = stack.enter_context(obj.access(**kwargs))
            yield accessed_obj
        elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            # Recursively flatten sequences (but not strings)
            yield from _flatten_and_access(tuple(obj), stack, kwargs)
        else:
            # Yield non-column objects unchanged (scalars, None, primitives, etc.)
            yield obj


class _AccessColumnsStack(ExitStack):
    """ExitStack subclass that returns all input objects on __enter__."""

    __slots__ = ("_objects_generator",)

    def __init__(self, objects_generator: Any) -> None:
        super().__init__()
        self._objects_generator = objects_generator

    def __enter__(self) -> tuple[Any, ...]:  # type: ignore[override]
        super().__enter__()
        # Consume generator and enter contexts here, not in __init__
        # _flatten_and_access will call enter_context on this stack
        # while yielding objects - consume and return directly
        return tuple(self._objects_generator)


def access_columns(
    *objs: Any,
    **kwargs: Any,
) -> _AccessColumnsStack:
    """Context manager to access multiple columns simultaneously.

    Simplifies the common pattern of using ExitStack to manage column access
    contexts. Automatically enters access contexts for column objects while
    passing through non-column objects (e.g., plc.Scalar, None, primitives)
    unchanged. Flattens nested sequences. Forwards kwargs to the access contexts.

    Returns
    -------
    Context manager that yields a tuple of all input objects in order. Column
    objects are replaced with their accessed versions, while non-column objects
    are returned unchanged.

    Examples
    --------
    >>> with access_columns(col1, col2, mode="read", scope="internal") as (c1, c2):
    ...     result = plc.operation(c1.plc_column, c2.plc_column)

    >>> # Works with mixed column and scalar inputs
    >>> with access_columns(col, scalar, mode="read", scope="internal") as (c, s):
    ...     result = plc.operation(c.plc_column, s)
    """
    # Create stack first, then assign generator
    # (can't use stack in generator before it exists)
    stack = object.__new__(_AccessColumnsStack)
    ExitStack.__init__(stack)
    stack._objects_generator = _flatten_and_access(objs, stack, kwargs)

    return stack
