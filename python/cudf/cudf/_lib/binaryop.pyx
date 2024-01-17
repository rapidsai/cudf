# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.column cimport Column

from cudf._lib.scalar import as_device_scalar

from cudf._lib.scalar cimport DeviceScalar

from cudf._lib import pylibcudf
from cudf._lib.types import SUPPORTED_NUMPY_TO_LIBCUDF_TYPES

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport data_type, type_id
from cudf._lib.types cimport dtype_to_pylibcudf_type, underlying_type_t_type_id

from cudf.core.buffer import acquire_spill_lock

cimport cudf._lib.cpp.binaryop as cpp_binaryop

import cudf

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
    if op not in {"INT_POW", "NULL_EQUALS"}:
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
            pylibcudf.binaryop.BinaryOperator[op.upper()],
            dtype_to_pylibcudf_type(dtype),
        )
    )


@acquire_spill_lock()
def binaryop_udf(Column lhs, Column rhs, udf_ptx, dtype):
    """
    Apply a user-defined binary operator (a UDF) defined in `udf_ptx` on
    the two input columns `lhs` and `rhs`. The output type of the UDF
    has to be specified in `dtype`, a numpy data type.
    Currently ONLY int32, int64, float32 and float64 are supported.
    """
    cdef column_view c_lhs = lhs.view()
    cdef column_view c_rhs = rhs.view()

    cdef type_id tid = (
        <type_id> (
            <underlying_type_t_type_id> (
                SUPPORTED_NUMPY_TO_LIBCUDF_TYPES[cudf.dtype(dtype)]
            )
        )
    )
    cdef data_type c_dtype = data_type(tid)

    cdef string cpp_str = udf_ptx.encode("UTF-8")

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_binaryop.binary_operation(
                c_lhs,
                c_rhs,
                cpp_str,
                c_dtype
            )
        )

    return Column.from_unique_ptr(move(c_result))
