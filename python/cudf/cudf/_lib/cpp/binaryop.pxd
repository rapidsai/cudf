# Copyright (c) 2020-2022, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.types cimport data_type


cdef extern from "cudf/binaryop.hpp" namespace "cudf" nogil:
    ctypedef enum binary_operator:
        ADD "cudf::binary_operator::ADD"
        SUB "cudf::binary_operator::SUB"
        MUL "cudf::binary_operator::MUL"
        DIV "cudf::binary_operator::DIV"
        TRUE_DIV "cudf::binary_operator::TRUE_DIV"
        FLOOR_DIV "cudf::binary_operator::FLOOR_DIV"
        MOD "cudf::binary_operator::MOD"
        PYMOD "cudf::binary_operator::PYMOD"
        POW "cudf::binary_operator::POW"
        INT_POW "cudf::binary_operator::INT_POW"
        EQUAL "cudf::binary_operator::EQUAL"
        NOT_EQUAL "cudf::binary_operator::NOT_EQUAL"
        LESS "cudf::binary_operator::LESS"
        GREATER "cudf::binary_operator::GREATER"
        LESS_EQUAL "cudf::binary_operator::LESS_EQUAL"
        GREATER_EQUAL "cudf::binary_operator::GREATER_EQUAL"
        NULL_EQUALS "cudf::binary_operator::NULL_EQUALS"
        BITWISE_AND "cudf::binary_operator::BITWISE_AND"
        BITWISE_OR "cudf::binary_operator::BITWISE_OR"
        BITWISE_XOR "cudf::binary_operator::BITWISE_XOR"
        LOGICAL_AND "cudf::binary_operator::LOGICAL_AND"
        LOGICAL_OR "cudf::binary_operator::LOGICAL_OR"
        GENERIC_BINARY "cudf::binary_operator::GENERIC_BINARY"

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

    unique_ptr[column] jit_binary_operation \
        "cudf::jit::binary_operation" (
        const column_view& lhs,
        const column_view& rhs,
        binary_operator op,
        data_type output_type
    ) except +

    unique_ptr[column] jit_binary_operation \
        "cudf::jit::binary_operation" (
        const column_view& lhs,
        const scalar& rhs,
        binary_operator op,
        data_type output_type
    ) except +

    unique_ptr[column] jit_binary_operation \
        "cudf::jit::binary_operation" (
        const scalar& lhs,
        const column_view& rhs,
        binary_operator op,
        data_type output_type
    ) except +
