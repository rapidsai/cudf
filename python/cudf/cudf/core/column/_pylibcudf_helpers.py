# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Helper functions that wrap common pylibcudf operations for column classes.

These functions provide efficient implementations of commonly-used patterns
that would otherwise require multiple column API calls. By going directly to
pylibcudf, they avoid creating temporary intermediate column objects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pylibcudf as plc

if TYPE_CHECKING:
    from cudf.core.buffer import Buffer
    from cudf.core.column.column import ColumnBase
    from cudf.core.column.string import StringColumn

__all__ = [
    "all_strings_match_type",
    "create_non_null_mask",
    "reduce_boolean_column",
]


def all_strings_match_type(
    column: StringColumn,
    type_check: Literal["integer", "float"],
) -> bool:
    """Check if all non-null strings in a column match a type pattern.

    This is more efficient than calling column.is_X().all() because it
    avoids creating intermediate column objects.

    Parameters
    ----------
    column : StringColumn
        The string column to check
    type_check : {"integer", "float"}
        The type pattern to check for

    Returns
    -------
    bool
        True if all non-null strings match the pattern

    Examples
    --------
    Instead of:

        if not col.is_integer().all():
            raise ValueError(...)

    Use:

        from cudf.core.column._pylibcudf_helpers import all_strings_match_type
        if not all_strings_match_type(col, "integer"):
            raise ValueError(...)
    """
    with column.access(mode="read", scope="internal"):
        # Get boolean column for type check
        if type_check == "integer":
            bool_plc = plc.strings.convert.convert_integers.is_integer(
                column.plc_column
            )
        elif type_check == "float":
            bool_plc = plc.strings.convert.convert_floats.is_float(
                column.plc_column
            )
        else:
            raise ValueError(f"Unknown type_check: {type_check}")

        # Reduce directly without creating intermediate column
        agg = plc.aggregation.all()
        result_scalar = plc.reduce.reduce(
            bool_plc,
            agg,
            plc.types.DataType(plc.types.TypeId.BOOL8),
        )

        # Extract boolean value from scalar
        result = result_scalar.to_py()
        assert isinstance(result, bool)
        return result


def create_non_null_mask(column: ColumnBase) -> Buffer:
    """Create a bitmask where only non-null elements are set to valid.

    This is equivalent to column.notnull().fillna(False).as_mask()
    but more efficient by going directly through pylibcudf.

    Note: This function is not currently used but kept for potential
    future optimizations.

    Parameters
    ----------
    column : ColumnBase
        The column to create mask from

    Returns
    -------
    Buffer
        A bitmask buffer

    Examples
    --------
    Instead of:

        mask = col.notnull().fillna(False).as_mask()

    Use:

        from cudf.core.column._pylibcudf_helpers import create_non_null_mask
        mask = create_non_null_mask(col)
    """
    from cudf.core.buffer import as_buffer

    with column.access(mode="read", scope="internal"):
        # Get validity directly
        bool_plc = plc.unary.is_valid(column.plc_column)
        # Convert to bitmask - returns DeviceBuffer
        mask_dbuffer = plc.transform.bools_to_mask(bool_plc)
        # Wrap in Buffer
        return as_buffer(data=mask_dbuffer)


def reduce_boolean_column(
    column: ColumnBase, operation: Literal["all", "any"]
) -> bool:
    """Reduce a boolean column to a single bool value.

    This is more efficient than creating a NumericalColumn and calling
    .all() or .any() on it.

    Parameters
    ----------
    column : ColumnBase
        The boolean column to reduce (or a column that can be viewed as boolean)
    operation : {"all", "any"}
        The reduction operation

    Returns
    -------
    bool
        The result of the reduction

    Examples
    --------
    Instead of:

        if col.is_integer().all():
            ...

    Use:

        from cudf.core.column._pylibcudf_helpers import reduce_boolean_column
        bool_col = col.is_integer()
        if reduce_boolean_column(bool_col, "all"):
            ...

    Or better yet, use all_strings_match_type() directly for string type checks.
    """
    with column.access(mode="read", scope="internal"):
        if operation == "all":
            agg = plc.aggregation.all()
        elif operation == "any":
            agg = plc.aggregation.any()
        else:
            raise ValueError(f"Unknown operation: {operation}")

        result_scalar = plc.reduce.reduce(
            column.plc_column,
            agg,
            plc.types.DataType(plc.types.TypeId.BOOL8),
        )
        result = result_scalar.to_py()
        assert isinstance(result, bool)
        return result
