# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np
from enum import IntEnum

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from cudf._lib.binaryop cimport underlying_type_t_binary_operator
from cudf._lib.column cimport Column
from cudf._lib.replace import replace_nulls
from cudf._lib.scalar import as_scalar
from cudf._lib.scalar cimport Scalar
from cudf._lib.types import np_to_cudf_types
from cudf._lib.types cimport underlying_type_t_type_id

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport (
    data_type,
    type_id,
)

from cudf.utils.dtypes import is_string_dtype

from cudf._lib.cpp.binaryop cimport binary_operator
cimport cudf._lib.cpp.binaryop as cpp_binaryop


class BinaryOperation(IntEnum):
    ADD = (
        <underlying_type_t_binary_operator> binary_operator.ADD
    )
    SUB = (
        <underlying_type_t_binary_operator> binary_operator.SUB
    )
    MUL = (
        <underlying_type_t_binary_operator> binary_operator.MUL
    )
    DIV = (
        <underlying_type_t_binary_operator> binary_operator.DIV
    )
    TRUEDIV = (
        <underlying_type_t_binary_operator> binary_operator.TRUE_DIV
    )
    FLOORDIV = (
        <underlying_type_t_binary_operator> binary_operator.FLOOR_DIV
    )
    MOD = (
        <underlying_type_t_binary_operator> binary_operator.PYMOD
    )
    POW = (
        <underlying_type_t_binary_operator> binary_operator.POW
    )
    EQ = (
        <underlying_type_t_binary_operator> binary_operator.EQUAL
    )
    NE = (
        <underlying_type_t_binary_operator> binary_operator.NOT_EQUAL
    )
    LT = (
        <underlying_type_t_binary_operator> binary_operator.LESS
    )
    GT = (
        <underlying_type_t_binary_operator> binary_operator.GREATER
    )
    LE = (
        <underlying_type_t_binary_operator> binary_operator.LESS_EQUAL
    )
    GE = (
        <underlying_type_t_binary_operator> binary_operator.GREATER_EQUAL
    )
    AND = (
        <underlying_type_t_binary_operator> binary_operator.BITWISE_AND
    )
    OR = (
        <underlying_type_t_binary_operator> binary_operator.BITWISE_OR
    )
    XOR = (
        <underlying_type_t_binary_operator> binary_operator.BITWISE_XOR
    )
    L_AND = (
        <underlying_type_t_binary_operator> binary_operator.LOGICAL_AND
    )
    L_OR = (
        <underlying_type_t_binary_operator> binary_operator.LOGICAL_OR
    )
    COALESCE = (
        <underlying_type_t_binary_operator> binary_operator.COALESCE
    )
    GENERIC_BINARY = (
        <underlying_type_t_binary_operator> binary_operator.GENERIC_BINARY
    )


cdef binaryop_v_v(Column lhs, Column rhs,
                  binary_operator c_op, data_type c_dtype):
    cdef column_view c_lhs = lhs.view()
    cdef column_view c_rhs = rhs.view()

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_binaryop.binary_operation(
                c_lhs,
                c_rhs,
                c_op,
                c_dtype
            )
        )

    return Column.from_unique_ptr(move(c_result))


cdef binaryop_v_s(Column lhs, Scalar rhs,
                  binary_operator c_op, data_type c_dtype):
    cdef column_view c_lhs = lhs.view()
    cdef scalar* c_rhs = rhs.c_value.get()

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_binaryop.binary_operation(
                c_lhs,
                c_rhs[0],
                c_op,
                c_dtype
            )
        )

    return Column.from_unique_ptr(move(c_result))


cdef binaryop_s_v(Scalar lhs, Column rhs,
                  binary_operator c_op, data_type c_dtype):
    cdef scalar* c_lhs = lhs.c_value.get()
    cdef column_view c_rhs = rhs.view()

    cdef unique_ptr[column] c_result

    with nogil:
        c_result = move(
            cpp_binaryop.binary_operation(
                c_lhs[0],
                c_rhs,
                c_op,
                c_dtype
            )
        )

    return Column.from_unique_ptr(move(c_result))


def handle_null_for_string_column(Column input_col, op):
    if op in ('eq', 'lt', 'le', 'gt', 'ge'):
        return replace_nulls(input_col, False)

    elif op == 'ne':
        return replace_nulls(input_col, True)

    # Nothing needs to be done
    return input_col


def binaryop(lhs, rhs, op, dtype):
    """
    Dispatches a binary op call to the appropriate libcudf function:
    """
    op = BinaryOperation[op.upper()]
    cdef binary_operator c_op = <binary_operator> (
        <underlying_type_t_binary_operator> op
    )
    cdef type_id tid = (
        <type_id> (
            <underlying_type_t_type_id> (
                np_to_cudf_types[np.dtype(dtype)]
            )
        )
    )

    cdef data_type c_dtype = data_type(tid)

    if isinstance(lhs, Scalar) or np.isscalar(lhs) or lhs is None:

        is_string_col = is_string_dtype(rhs.dtype)
        s_lhs = as_scalar(lhs, dtype=rhs.dtype if lhs is None else None)
        result = binaryop_s_v(
            s_lhs,
            rhs,
            c_op,
            c_dtype
        )

    elif isinstance(rhs, Scalar) or np.isscalar(rhs) or rhs is None:
        is_string_col = is_string_dtype(lhs.dtype)
        s_rhs = as_scalar(rhs, dtype=lhs.dtype if rhs is None else None)
        result = binaryop_v_s(
            lhs,
            s_rhs,
            c_op,
            c_dtype
        )

    else:
        is_string_col = is_string_dtype(lhs.dtype)
        result = binaryop_v_v(
            lhs,
            rhs,
            c_op,
            c_dtype
        )

    if is_string_col is True:
        return handle_null_for_string_column(result, op.name.lower())
    else:
        return result


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
                np_to_cudf_types[np.dtype(dtype)]
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
