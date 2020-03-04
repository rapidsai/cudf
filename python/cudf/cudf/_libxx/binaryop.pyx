# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np
from enum import IntEnum

from libcpp.memory cimport unique_ptr

from cudf._libxx.binaryop cimport underlying_type_t_binary_operator
from cudf._libxx.column cimport Column
from cudf._libxx.move cimport move
from cudf._libxx.types import np_to_cudf_types

from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.types cimport (
    data_type,
    type_id,
)

cimport cudf._libxx.cpp.binaryop as cpp_binaryop


class BinaryOperation(IntEnum):
    ADD = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.ADD
    )
    SUB = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.SUB
    )
    MUL = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.MUL
    )
    DIV = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.DIV
    )
    TRUE_DIV = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.TRUE_DIV
    )
    FLOOR_DIV = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.FLOOR_DIV
    )
    MOD = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.MOD
    )
    PYMOD = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.PYMOD
    )
    POW = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.POW
    )
    EQUAL = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.EQUAL
    )
    NOT_EQUAL = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.NOT_EQUAL
    )
    LESS = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.LESS
    )
    GREATER = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.GREATER
    )
    LESS_EQUAL = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.LESS_EQUAL
    )
    GREATER_EQUAL = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.GREATER_EQUAL
    )
    BITWISE_AND = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.BITWISE_AND
    )
    BITWISE_OR = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.BITWISE_OR
    )
    BITWISE_XOR = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.BITWISE_XOR
    )
    LOGICAL_AND = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.LOGICAL_AND
    )
    LOGICAL_OR = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.LOGICAL_OR
    )
    COALESCE = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.COALESCE
    )
    GENERIC_BINARY = (
        <underlying_type_t_binary_operator> cpp_binaryop.binary_operator.GENERIC_BINARY
    )


def binaryop(Column lhs, Column rhs, object op, object dtype):
    """
    Dispatches a binary op call to the appropriate libcudf function:
    """
    cdef column_view c_lhs = lhs.view()
    cdef column_view c_rhs = rhs.view()
    cdef cpp_binaryop.binary_operator c_op = \
        <cpp_binaryop.binary_operator>(
            <underlying_type_t_binary_operator> op
        )
    cdef type_id tid = np_to_cudf_types[np.dtype(dtype)]
    cdef data_type c_dtype = data_type(tid)

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
