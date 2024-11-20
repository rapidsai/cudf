# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf._lib.types import dtype_to_pylibcudf_type
from cudf.api.types import is_decimal_dtype
from cudf.core.buffer import acquire_spill_lock

if TYPE_CHECKING:
    from cudf._typing import Dtype
    from cudf.core.column import ColumnBase


@acquire_spill_lock()
def unary_operation(
    col: ColumnBase, op: plc.unary.UnaryOperator
) -> ColumnBase:
    return type(col).from_pylibcudf(
        plc.unary.unary_operation(col.to_pylibcudf(mode="read"), op)
    )


@acquire_spill_lock()
def is_null(col: ColumnBase) -> ColumnBase:
    return type(col).from_pylibcudf(
        plc.unary.is_null(col.to_pylibcudf(mode="read"))
    )


@acquire_spill_lock()
def is_valid(col: ColumnBase) -> ColumnBase:
    return type(col).from_pylibcudf(
        plc.unary.is_valid(col.to_pylibcudf(mode="read"))
    )


@acquire_spill_lock()
def cast(col: ColumnBase, dtype: Dtype) -> ColumnBase:
    result = type(col).from_pylibcudf(
        plc.unary.cast(
            col.to_pylibcudf(mode="read"), dtype_to_pylibcudf_type(dtype)
        )
    )

    if is_decimal_dtype(result.dtype):
        result.dtype.precision = dtype.precision  # type: ignore[union-attr]
    return result


@acquire_spill_lock()
def is_nan(col: ColumnBase) -> ColumnBase:
    return type(col).from_pylibcudf(
        plc.unary.is_nan(col.to_pylibcudf(mode="read"))
    )


@acquire_spill_lock()
def is_non_nan(col: ColumnBase) -> ColumnBase:
    return type(col).from_pylibcudf(
        plc.unary.is_not_nan(col.to_pylibcudf(mode="read"))
    )
