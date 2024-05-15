# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from cudf._lib.column cimport Column
from cudf._lib.scalar cimport DeviceScalar
from cudf._lib.types cimport dtype_to_pylibcudf_type

from cudf._lib import pylibcudf
from cudf._lib.scalar import as_device_scalar
from cudf.core.buffer import acquire_spill_lock

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


@acquire_spill_lock()
def binaryop(lhs, rhs, op, dtype):
    """
    Dispatches a binary op call to the appropriate libcudf function:
    """
    # TODO: Shouldn't have to keep special-casing. We need to define a separate
    # pipeline for libcudf binops that don't map to Python binops.
    if op not in {"INT_POW", "NULL_EQUALS", "NULL_NOT_EQUALS"}:
        op = op[2:-2]
    op = op.upper()
    op = _op_map.get(op, op)

    return Column.from_pylibcudf(
        # Check if the dtype args are desirable here.
        pylibcudf.binaryop.binary_operation(
            lhs.to_pylibcudf(mode="read") if isinstance(lhs, Column)
            else (
                <DeviceScalar> as_device_scalar(
                    lhs, dtype=rhs.dtype if lhs is None else None
                )
            ).c_value,
            rhs.to_pylibcudf(mode="read") if isinstance(rhs, Column)
            else (
                <DeviceScalar> as_device_scalar(
                    rhs, dtype=lhs.dtype if rhs is None else None
                )
            ).c_value,
            pylibcudf.binaryop.BinaryOperator[op],
            dtype_to_pylibcudf_type(dtype),
        )
    )
