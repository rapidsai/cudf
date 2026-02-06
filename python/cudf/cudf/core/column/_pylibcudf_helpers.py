# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Helper functions that wrap common pylibcudf operations for column classes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import pylibcudf as plc

if TYPE_CHECKING:
    from cudf.core.column import ColumnBase, StringColumn

__all__ = [
    "string_is_float",
    "string_is_int",
]


def _all_strings_match_type(
    column: ColumnBase,
    type_check: Literal["integer", "float"],
) -> bool:
    """Check if all non-null strings in a column are integers or floats.

    Instead of `col.is_integer().all()`, use `string_is_int(col)` so that we can avoid
    creating an intermediate boolean ColumnBase and going through the ColumnBase.create
    loop two extra times.
    """
    string_column = cast("StringColumn", column)
    with string_column.access(mode="read", scope="internal"):
        assert type_check in ("integer", "float"), (
            f"Invalid type_check: {type_check}"
        )
        if type_check == "integer":
            bool_plc = plc.strings.convert.convert_integers.is_integer(
                string_column.plc_column
            )
        else:
            bool_plc = plc.strings.convert.convert_floats.is_float(
                string_column.plc_column
            )

        result_scalar = plc.reduce.reduce(
            bool_plc,
            plc.aggregation.all(),
            plc.types.DataType(plc.types.TypeId.BOOL8),
        )

        result = result_scalar.to_py()
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"
        return result


def string_is_int(column: ColumnBase) -> bool:
    """Check if all non-null strings in a column are integers."""
    return _all_strings_match_type(column, "integer")


def string_is_float(column: ColumnBase) -> bool:
    """Check if all non-null strings in a column are floats."""
    return _all_strings_match_type(column, "float")
