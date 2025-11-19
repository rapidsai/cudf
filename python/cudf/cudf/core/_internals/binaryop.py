# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc

from cudf.core.buffer import acquire_spill_lock
from cudf.core.column import ColumnBase
from cudf.utils.dtypes import dtype_to_pylibcudf_type

if TYPE_CHECKING:
    from cudf._typing import DtypeObj


@acquire_spill_lock()
def binaryop(
    lhs: ColumnBase | plc.Scalar,
    rhs: ColumnBase | plc.Scalar,
    op: str,
    dtype: DtypeObj,
) -> ColumnBase:
    """
    Dispatches a binary op call to the appropriate libcudf function:
    """
    # TODO: Shouldn't have to keep special-casing. We need to define a separate
    # pipeline for libcudf binops that don't map to Python binops.
    if op not in {"INT_POW", "NULL_EQUALS", "NULL_NOT_EQUALS"}:
        op = op[2:-2]
    # Map pandas operation names to pylibcudf operation names.
    _op_map = {
        "TRUEDIV": "TRUE_DIV",
        "FLOORDIV": "FLOOR_DIV",
        "MOD": "PYMOD",
        "EQ": "EQUAL",
        "NE": "NOT_EQUAL",
        "LT": "LESS",
        "GT": "GREATER",
        "LE": "LESS_EQUAL",
        "GE": "GREATER_EQUAL",
        "AND": "BITWISE_AND",
        "OR": "BITWISE_OR",
        "XOR": "BITWISE_XOR",
        "L_AND": "LOGICAL_AND",
        "L_OR": "LOGICAL_OR",
    }
    op = op.upper()
    op = _op_map.get(op, op)

    return ColumnBase.from_pylibcudf(
        plc.binaryop.binary_operation(
            lhs.to_pylibcudf(mode="read")
            if isinstance(lhs, ColumnBase)
            else lhs,
            rhs.to_pylibcudf(mode="read")
            if isinstance(rhs, ColumnBase)
            else rhs,
            plc.binaryop.BinaryOperator[op],
            dtype_to_pylibcudf_type(dtype),
        )
    )._with_type_metadata(dtype)
