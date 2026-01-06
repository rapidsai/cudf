# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for column operations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from typing import Any


def _flatten_to_columns(objs: tuple[Any, ...]) -> Iterator[Any]:
    """Recursively flatten varargs to column objects."""
    for obj in objs:
        # Check if object has .access() method (duck typing)
        if (access := getattr(obj, "access", None)) is not None and callable(
            access
        ):
            yield obj
        elif isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            # Recursively flatten sequences (but not strings)
            yield from _flatten_to_columns(tuple(obj))
        # Silently skip other objects


def access_columns(
    *objs: Any,
    **kwargs: Any,
) -> ExitStack:
    """Context manager to access multiple columns simultaneously.

    Simplifies the common pattern of using ExitStack to manage column access
    contexts. Automatically filters out non-column objects (e.g., plc.Scalar,
    None, primitives) and flattens nested sequences. Forwards kwargs to the called
    contexts.
    """
    stack = ExitStack()

    # Enter all column access contexts
    for col in _flatten_to_columns(objs):
        stack.enter_context(col.access(**kwargs))

    return stack
