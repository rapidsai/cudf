# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Helper functions that wrap common pylibcudf operations for column classes.

These functions provide efficient implementations of commonly-used patterns
that would otherwise require multiple column API calls, eliminating
intermediate column allocations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pylibcudf as plc

if TYPE_CHECKING:
    from cudf.core.column import ColumnBase

__all__ = [
    "all_strings_match_type",
    "fillna_bool_false",
    "isnull_including_nan",
    "notnull_excluding_nan",
    "reduce_boolean_column",
]


def all_strings_match_type(
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
        if not all_strings_match_type(col, "integer"):
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


def reduce_boolean_column(
    column: ColumnBase, operation: Literal["all", "any"]
) -> bool:
    """Reduce a boolean column to a single bool value.

    This is more efficient than creating a NumericalColumn and calling
    .all() or .any() on it when you already have a boolean column.

    Parameters
    ----------
    column : ColumnBase
        The boolean column to reduce
    operation : {"all", "any"}
        The reduction operation

    Returns
    -------
    bool
        The result of the reduction

    Examples
    --------
    Instead of:
        bool_col = col.is_integer()
        if bool_col.all():
            ...

    Use:
        if reduce_boolean_column(col.is_integer(), "all"):
            ...

    Or better yet, use all_strings_match_type() directly for string
    validation patterns.

    Notes
    -----
    This function is most useful when you need to perform a reduction
    on an already-computed boolean column. For string validation patterns,
    prefer all_strings_match_type() which combines the type check and
    reduction in one operation.
    """
    with column.access(mode="read", scope="internal"):
        if operation == "all":
            agg = plc.aggregation.all()
        elif operation == "any":
            agg = plc.aggregation.any()
        else:
            raise ValueError(
                f"Unknown operation: {operation}. Must be 'all' or 'any'."
            )

        result_scalar = plc.reduce.reduce(
            column.plc_column,
            agg,
            plc.types.DataType(plc.types.TypeId.BOOL8),
        )

        result = result_scalar.to_py()
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"
        return result


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


def isnull_including_nan(column: ColumnBase) -> ColumnBase:
    """Check for null values including NaN for float columns.

    This is more efficient than calling is_null() | isnan() because it
    combines both checks in a single operation.

    Parameters
    ----------
    column : ColumnBase
        Numerical column to check for nulls

    Returns
    -------
    ColumnBase
        Boolean column with True where values are null or NaN

    Examples
    --------
    Instead of:

        result = is_null_col | isnan_col  # Two column allocations

    Use:

        from cudf.core.column._pylibcudf_helpers import isnull_including_nan
        result = isnull_including_nan(col)

    Notes
    -----
    For float columns, both null mask nulls and NaN values are considered null.
    For non-float columns, this is equivalent to is_null().
    """
    import numpy as np

    from cudf.core.column.column import ColumnBase

    with column.access(mode="read", scope="internal"):
        # Get null mask
        is_null_plc = plc.unary.is_null(column.plc_column)

        # For float types, also check for NaN
        if column.dtype.kind == "f":
            is_nan_plc = plc.unary.is_nan(column.plc_column)
            # Combine using bitwise OR
            result_plc = plc.binaryop.binary_operation(
                is_null_plc,
                is_nan_plc,
                plc.binaryop.BinaryOperator.BITWISE_OR,
                plc.types.DataType(plc.types.TypeId.BOOL8),
            )
            return ColumnBase.from_pylibcudf(result_plc)
        else:
            return ColumnBase.create(is_null_plc, np.dtype(np.bool_))


def notnull_excluding_nan(column: ColumnBase) -> ColumnBase:
    """Check for non-null values excluding NaN for float columns.

    This is more efficient than calling is_valid() & notnan() because it
    combines both checks in a single operation.

    Parameters
    ----------
    column : ColumnBase
        Numerical column to check for non-nulls

    Returns
    -------
    ColumnBase
        Boolean column with True where values are not null and not NaN

    Examples
    --------
    Instead of:

        result = is_valid_col & notnan_col  # Two column allocations

    Use:

        from cudf.core.column._pylibcudf_helpers import notnull_excluding_nan
        result = notnull_excluding_nan(col)

    Notes
    -----
    For float columns, both null mask nulls and NaN values are considered null.
    For non-float columns, this is equivalent to is_valid().
    """
    import numpy as np

    from cudf.core.column.column import ColumnBase

    with column.access(mode="read", scope="internal"):
        # Get validity mask
        is_valid_plc = plc.unary.is_valid(column.plc_column)

        # For float types, also check for non-NaN
        if column.dtype.kind == "f":
            is_not_nan_plc = plc.unary.is_not_nan(column.plc_column)
            # Combine using bitwise AND
            result_plc = plc.binaryop.binary_operation(
                is_valid_plc,
                is_not_nan_plc,
                plc.binaryop.BinaryOperator.BITWISE_AND,
                plc.types.DataType(plc.types.TypeId.BOOL8),
            )
            return ColumnBase.from_pylibcudf(result_plc)
        else:
            return ColumnBase.create(is_valid_plc, np.dtype(np.bool_))
