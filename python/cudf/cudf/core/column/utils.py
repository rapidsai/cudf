# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for column operations."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from typing import Any


def access_columns(
    *objs: Any,
    mode: str,
    scope: str,
) -> ExitStack:
    """
    Context manager to access multiple columns simultaneously.

    Simplifies the common pattern of using ExitStack to manage column access
    contexts. Automatically filters out non-column objects (e.g., plc.Scalar,
    None, primitives) and flattens nested sequences.

    Parameters
    ----------
    *objs : ColumnBase | Iterable[ColumnBase] | Any
        Column objects or sequences of columns to access. Can be:
        - Individual columns: access_columns(col1, col2, col3, mode="read", scope="internal")
        - Sequences: access_columns(*column_list, mode="read", scope="internal")
        - Mixed: access_columns(col1, *more_cols, col2, mode="read", scope="internal")
        Non-column objects are silently skipped.
    mode : str
        Access mode for copy-on-write behavior (required).
    scope : str
        Spill scope for SpillableBuffer (required).

    Returns
    -------
    ExitStack
        A context manager that manages access contexts for all columns.

    Examples
    --------
    >>> # Fixed columns
    >>> with access_columns(self, other, mode="read", scope="internal"):
    ...     result = plc.operation(self.plc_column, other.plc_column)

    >>> # Variable columns
    >>> with access_columns(*column_list, mode="read", scope="internal"):
    ...     plc_cols = [c.plc_column for c in column_list]
    ...     result = plc.concatenate.concatenate(plc_cols)

    >>> # Automatically skips non-columns
    >>> with access_columns(self, other, boolean_mask, mode="read", scope="internal"):
    ...     # Works even if other is plc.Scalar
    ...     other_col = other if isinstance(other, plc.Scalar) else other.plc_column
    ...     result = plc.copying.copy_if_else(
    ...         self.plc_column, other_col, boolean_mask.plc_column
    ...     )
    """
    stack = ExitStack()

    def _flatten_to_columns(objs: tuple[Any, ...]) -> Iterator[Any]:
        """Recursively flatten varargs to column objects."""
        for obj in objs:
            # Check if object has .access() method (duck typing)
            if hasattr(obj, "access") and callable(
                getattr(obj, "access", None)
            ):
                yield obj
            elif isinstance(obj, Iterable) and not isinstance(
                obj, (str, bytes)
            ):
                # Recursively flatten sequences (but not strings)
                yield from _flatten_to_columns(tuple(obj))
            # Silently skip: Scalars, None, primitives, etc.

    # Enter all column access contexts
    for col in _flatten_to_columns(objs):
        stack.enter_context(col.access(mode=mode, scope=scope))

    return stack
