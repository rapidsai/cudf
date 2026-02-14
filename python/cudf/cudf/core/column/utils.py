# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for column operations."""

from __future__ import annotations

import inspect
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


def _access_object(obj: Any, stack: ExitStack, kwargs: dict[str, Any]) -> Any:
    access = getattr(obj, "access", None)
    if access is not None and callable(access):
        return stack.enter_context(
            cast(AbstractContextManager[Any], access(**kwargs))
        )
    if isinstance(obj, list):
        return [_access_object(item, stack, kwargs) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_access_object(item, stack, kwargs) for item in obj)
    return obj


def _collect_column_dtypes(obj: Any) -> list[Any]:
    access = getattr(obj, "access", None)
    if access is not None and callable(access):
        return [obj.dtype]
    if isinstance(obj, list):
        return [
            dtype for item in obj for dtype in _collect_column_dtypes(item)
        ]
    if isinstance(obj, tuple):
        return [
            dtype for item in obj for dtype in _collect_column_dtypes(item)
        ]
    return []


def _columns_to_plc_column(obj: Any) -> Any:
    plc_column = getattr(obj, "plc_column", None)
    if plc_column is not None:
        return plc_column
    if isinstance(obj, list):
        return [_columns_to_plc_column(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_columns_to_plc_column(item) for item in obj)
    return obj


def columns_to_plc_columns(*args: Any) -> tuple[Any, ...]:
    return tuple(_columns_to_plc_column(arg) for arg in args)


def plc_column_op(
    plc_fn: Callable[..., Any] | None = None,
    *,
    column_args: tuple[str, ...],
    dtype_policy: Callable[[list[Any], Any], Any] | None = None,
    mode: Literal["read", "write"] = "read",
    scope: Literal["internal", "external"] = "internal",
    returns_plc_fn: bool = False,
    wrap_output: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            access_kwargs = {"mode": mode, "scope": scope}
            column_values = [bound.arguments[name] for name in column_args]
            with ExitStack() as stack:
                accessed = {
                    name: _access_object(
                        bound.arguments[name], stack, access_kwargs
                    )
                    for name in column_args
                }
                bound.arguments.update(accessed)
                result = func(*bound.args, **bound.kwargs)

                if returns_plc_fn:
                    if not isinstance(result, tuple) or len(result) < 2:
                        raise TypeError(
                            "Expected (plc_fn, args, kwargs) from decorated function"
                        )
                    resolved_plc_fn = result[0]
                    plc_args = result[1]
                    plc_kwargs = result[2] if len(result) > 2 else {}
                else:
                    if plc_fn is None:
                        raise TypeError(
                            "plc_fn must be provided when returns_plc_fn is False"
                        )
                    resolved_plc_fn = plc_fn
                    if (
                        isinstance(result, tuple)
                        and len(result) == 2
                        and isinstance(result[1], dict)
                    ):
                        plc_args, plc_kwargs = result
                    else:
                        plc_args, plc_kwargs = result, {}

                if not isinstance(plc_args, (list, tuple)):
                    plc_args = (plc_args,)
                plc_args = columns_to_plc_columns(*plc_args)
                plc_kwargs = {
                    key: _columns_to_plc_column(value)
                    for key, value in plc_kwargs.items()
                }

                plc_result = resolved_plc_fn(*plc_args, **plc_kwargs)
                if not wrap_output:
                    return plc_result

                if dtype_policy is None:
                    raise TypeError(
                        "dtype_policy must be provided when wrap_output is True"
                    )
                dtypes = [
                    dtype
                    for value in column_values
                    for dtype in _collect_column_dtypes(value)
                ]
                output_dtype = dtype_policy(dtypes, bound)
                from cudf.core.column.column import ColumnBase

                return ColumnBase.create(plc_result, dtype=output_dtype)

        return wrapper

    return decorator
