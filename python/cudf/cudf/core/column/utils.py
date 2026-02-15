# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for column operations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, ExitStack
from functools import wraps
from typing import Any, Literal, cast


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
            accessed_obj = stack.enter_context(
                cast(AbstractContextManager[Any], obj.access(**kwargs))
            )
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


class PylibcudfFunction:
    def __init__(
        self,
        pylibcudf_function: Callable[..., Any],
        *,
        dtype_policy: Callable[[list[Any]], Any],
        mode: Literal["read", "write"] = "read",
        scope: Literal["internal", "external"] = "internal",
    ) -> None:
        self._pylibcudf_function = pylibcudf_function
        self._dtype_policy = dtype_policy
        self._mode = mode
        self._scope = scope

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        from cudf.core.column.column import ColumnBase

        columns = [arg for arg in args if isinstance(arg, ColumnBase)]
        columns.extend(
            value for value in kwargs.values() if isinstance(value, ColumnBase)
        )
        if columns:
            with access_columns(
                *columns, mode=self._mode, scope=self._scope
            ) as accessed:
                accessed_iter = iter(accessed)
                args = tuple(
                    next(accessed_iter) if isinstance(arg, ColumnBase) else arg
                    for arg in args
                )
                kwargs = {
                    key: (
                        next(accessed_iter)
                        if isinstance(value, ColumnBase)
                        else value
                    )
                    for key, value in kwargs.items()
                }

        plc_args = tuple(
            arg.plc_column if isinstance(arg, ColumnBase) else arg
            for arg in args
        )
        plc_kwargs = {
            key: (value.plc_column if isinstance(value, ColumnBase) else value)
            for key, value in kwargs.items()
        }
        plc_result = self._pylibcudf_function(*plc_args, **plc_kwargs)
        output_dtype = self._dtype_policy([column.dtype for column in columns])
        return ColumnBase.create(plc_result, dtype=output_dtype)


def pylibcudf_op(
    pylibcudf_function: Callable[..., Any],
    *,
    dtype_policy: Callable[[list[Any]], Any],
    mode: Literal["read", "write"] = "read",
    scope: Literal["internal", "external"] = "internal",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        op = PylibcudfFunction(
            pylibcudf_function,
            dtype_policy=dtype_policy,
            mode=mode,
            scope=scope,
        )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return op(*args, **kwargs)

        return wrapper

    return decorator
