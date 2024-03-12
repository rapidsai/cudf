# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.types cimport data_type


cdef extern from "cudf/binaryop.hpp" namespace "cudf" nogil:
    cpdef enum class binary_operator(int32_t):
        ADD
        SUB
        MUL
        DIV
        TRUE_DIV
        FLOOR_DIV
        MOD
        PMOD
        PYMOD
        POW
        INT_POW
        LOG_BASE
        ATAN2
        SHIFT_LEFT
        SHIFT_RIGHT
        SHIFT_RIGHT_UNSIGNED
        BITWISE_AND
        BITWISE_OR
        BITWISE_XOR
        LOGICAL_AND
        LOGICAL_OR
        EQUAL
        NOT_EQUAL
        LESS
        GREATER
        LESS_EQUAL
        GREATER_EQUAL
        NULL_EQUALS
        NULL_MAX
        NULL_MIN
        GENERIC_BINARY
        NULL_LOGICAL_AND
        NULL_LOGICAL_OR
        INVALID_BINARY

    cdef unique_ptr[column] binary_operation (
        const scalar& lhs,
        const column_view& rhs,
        binary_operator op,
        data_type output_type
    ) except +

    cdef unique_ptr[column] binary_operation (
        const column_view& lhs,
        const scalar& rhs,
        binary_operator op,
        data_type output_type
    ) except +

    cdef unique_ptr[column] binary_operation (
        const column_view& lhs,
        const column_view& rhs,
        binary_operator op,
        data_type output_type
    ) except +

    cdef unique_ptr[column] binary_operation (
        const column_view& lhs,
        const column_view& rhs,
        const string& op,
        data_type output_type
    ) except +
