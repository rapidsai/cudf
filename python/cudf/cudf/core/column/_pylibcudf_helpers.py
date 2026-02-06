# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Helper functions that wrap common pylibcudf operations for column classes.

These functions bypass the need to create intermediate ColumnBases by operating directly
on pylibcudf columns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pylibcudf as plc

if TYPE_CHECKING:
    from cudf.core.column import ColumnBase

__all__ = [
    "fillna_bool_false",
    "fillna_numeric_zero",
    "string_is_float",
    "string_is_int",
]


def _all_strings_match_type(
    column: ColumnBase,
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
        if not string_is_int(col):
            raise ValueError(...)

    Notes
    -----
    This function combines the is_X() type check and the all() reduction
    into a single operation, eliminating the need to create an intermediate
    boolean column. The reduction can potentially short-circuit on the first
    False value, providing additional performance benefits.
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
            raise ValueError(
                f"Unknown type_check: {type_check}. "
                f"Must be 'integer' or 'float'."
            )

        # Reduce directly without creating intermediate column
        result_scalar = plc.reduce.reduce(
            bool_plc,
            plc.aggregation.all(),
            plc.types.DataType(plc.types.TypeId.BOOL8),
        )

        # Extract boolean value from scalar
        result = result_scalar.to_py()
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"
        return result


def string_is_int(column: ColumnBase) -> bool:
    """Check if all non-null strings in a column are integers."""
    return _all_strings_match_type(column, "integer")


def string_is_float(column: ColumnBase) -> bool:
    """Check if all non-null strings in a column are floats."""
    return _all_strings_match_type(column, "float")


def fillna_bool_false(column: ColumnBase) -> ColumnBase:
    """Fill null values in a boolean column with False.

    This is more efficient than calling column.fillna(False) because it
    avoids creating intermediate column objects by using pylibcudf's
    replace_nulls directly.

    Parameters
    ----------
    column : ColumnBase
        Boolean column with potential null values

    Returns
    -------
    ColumnBase
        Boolean column with nulls replaced by False

    Examples
    --------
    Instead of:

        result = (self.day == 1).fillna(False)

    Use:

        from cudf.core.column._pylibcudf_helpers import fillna_bool_false
        result = fillna_bool_false(self.day == 1)

    Notes
    -----
    This is particularly useful for datetime property methods that return
    boolean columns with nulls that should be treated as False (e.g.,
    is_month_start, is_year_end, etc.)
    """
    from cudf.core.column.column import ColumnBase

    with column.access(mode="read", scope="internal"):
        # Use pylibcudf replace_nulls with scalar False
        false_scalar = plc.Scalar.from_py(False)
        result_plc = plc.replace.replace_nulls(column.plc_column, false_scalar)
        return ColumnBase.from_pylibcudf(result_plc)


def fillna_numeric_zero(column: ColumnBase) -> ColumnBase:
    """Fill null values in a numeric column with 0.

    This is more efficient than calling column.fillna(0) because it
    avoids creating intermediate column objects by using pylibcudf's
    replace_nulls directly.

    Parameters
    ----------
    column : ColumnBase
        Numeric column with potential null values

    Returns
    -------
    ColumnBase
        Numeric column with nulls replaced by 0

    Examples
    --------
    Instead of:

        result = col.fillna(0)

    Use:

        from cudf.core.column._pylibcudf_helpers import fillna_numeric_zero
        result = fillna_numeric_zero(col)

    Notes
    -----
    This is particularly useful for numeric operations where null values
    should be treated as zero (e.g., in join helpers, groupby operations, etc.)
    """
    from cudf.core.column.column import ColumnBase
    from cudf.utils.dtypes import dtype_to_pylibcudf_type

    with column.access(mode="read", scope="internal"):
        # Use pylibcudf replace_nulls with a typed scalar zero
        zero_scalar = plc.Scalar.from_py(
            0, dtype_to_pylibcudf_type(column.dtype)
        )
        result_plc = plc.replace.replace_nulls(column.plc_column, zero_scalar)
        return ColumnBase.from_pylibcudf(result_plc)
