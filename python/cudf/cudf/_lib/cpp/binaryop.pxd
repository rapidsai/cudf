# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport (
    data_type
)

cdef extern from "cudf/binaryop.hpp" namespace "cudf" nogil:
    ctypedef enum binary_operator:
        ADD "cudf::binary_op::ADD"
        SUB "cudf::binary_op::SUB"
        MUL "cudf::binary_op::MUL"
        DIV "cudf::binary_op::DIV"
        TRUE_DIV "cudf::binary_op::TRUE_DIV"
        FLOOR_DIV "cudf::binary_op::FLOOR_DIV"
        MOD "cudf::binary_op::MOD"
        PYMOD "cudf::binary_op::PYMOD"
        POW "cudf::binary_op::POW"
        EQUAL "cudf::binary_op::EQUAL"
        NOT_EQUAL "cudf::binary_op::NOT_EQUAL"
        LESS "cudf::binary_op::LESS"
        GREATER "cudf::binary_op::GREATER"
        LESS_EQUAL "cudf::binary_op::LESS_EQUAL"
        GREATER_EQUAL "cudf::binary_op::GREATER_EQUAL"
        BITWISE_AND "cudf::binary_op::BITWISE_AND"
        BITWISE_OR "cudf::binary_op::BITWISE_OR"
        BITWISE_XOR "cudf::binary_op::BITWISE_XOR"
        LOGICAL_AND "cudf::binary_op::LOGICAL_AND"
        LOGICAL_OR "cudf::binary_op::LOGICAL_OR"
        COALESCE "cudf::binary_op::COALESCE"
        GENERIC_BINARY "cudf::binary_op::GENERIC_BINARY"

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
