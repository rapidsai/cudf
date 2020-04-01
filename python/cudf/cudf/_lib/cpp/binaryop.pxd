# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport (
    data_type
)

cdef extern from "cudf/binaryop.hpp" namespace "cudf::experimental" nogil:
    ctypedef enum binary_operator:
        ADD "cudf::experimental::binary_operator::ADD"
        SUB "cudf::experimental::binary_operator::SUB"
        MUL "cudf::experimental::binary_operator::MUL"
        DIV "cudf::experimental::binary_operator::DIV"
        TRUE_DIV "cudf::experimental::binary_operator::TRUE_DIV"
        FLOOR_DIV "cudf::experimental::binary_operator::FLOOR_DIV"
        MOD "cudf::experimental::binary_operator::MOD"
        PYMOD "cudf::experimental::binary_operator::PYMOD"
        POW "cudf::experimental::binary_operator::POW"
        EQUAL "cudf::experimental::binary_operator::EQUAL"
        NOT_EQUAL "cudf::experimental::binary_operator::NOT_EQUAL"
        LESS "cudf::experimental::binary_operator::LESS"
        GREATER "cudf::experimental::binary_operator::GREATER"
        LESS_EQUAL "cudf::experimental::binary_operator::LESS_EQUAL"
        GREATER_EQUAL "cudf::experimental::binary_operator::GREATER_EQUAL"
        BITWISE_AND "cudf::experimental::binary_operator::BITWISE_AND"
        BITWISE_OR "cudf::experimental::binary_operator::BITWISE_OR"
        BITWISE_XOR "cudf::experimental::binary_operator::BITWISE_XOR"
        LOGICAL_AND "cudf::experimental::binary_operator::LOGICAL_AND"
        LOGICAL_OR "cudf::experimental::binary_operator::LOGICAL_OR"
        COALESCE "cudf::experimental::binary_operator::COALESCE"
        GENERIC_BINARY "cudf::experimental::binary_operator::GENERIC_BINARY"

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
