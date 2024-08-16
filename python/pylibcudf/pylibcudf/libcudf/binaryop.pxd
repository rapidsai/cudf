# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.types cimport data_type


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
        NULL_NOT_EQUALS
        GENERIC_BINARY
        NULL_LOGICAL_AND
        NULL_LOGICAL_OR
        INVALID_BINARY

    cdef unique_ptr[column] binary_operation (
        const scalar& lhs,
        const column_view& rhs,
        binary_operator op,
        data_type output_type
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] binary_operation (
        const column_view& lhs,
        const scalar& rhs,
        binary_operator op,
        data_type output_type
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] binary_operation (
        const column_view& lhs,
        const column_view& rhs,
        binary_operator op,
        data_type output_type
    ) except +libcudf_exception_handler

    cdef unique_ptr[column] binary_operation (
        const column_view& lhs,
        const column_view& rhs,
        const string& op,
        data_type output_type
    ) except +libcudf_exception_handler

cdef extern from "cudf/binaryop.hpp" namespace "cudf::binops" nogil:
    cdef bool is_supported_operation(
        data_type output_type,
        data_type lhs_type,
        data_type rhs_type,
        binary_operator op
    ) except +libcudf_exception_handler
