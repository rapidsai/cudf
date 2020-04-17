# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr
from cudf._lib.cpp.column.column_view cimport (
    column_view
)
from cudf._lib.cpp.column.column cimport (
    column
)
from cudf._lib.cpp.types cimport (
    data_type
)

ctypedef int32_t underlying_type_t_unary_op


cdef extern from "cudf/unary.hpp" namespace "cudf::experimental" nogil:

    ctypedef enum unary_op:
        SIN "cudf::experimental::unary_op::SIN"
        COS "cudf::experimental::unary_op::COS"
        TAN "cudf::experimental::unary_op::TAN"
        ARCSIN "cudf::experimental::unary_op::ARCSIN"
        ARCCOS "cudf::experimental::unary_op::ARCCOS"
        ARCTAN "cudf::experimental::unary_op::ARCTAN"
        SINH "cudf::experimental::unary_op::SINH"
        COSH "cudf::experimental::unary_op::COSH"
        TANH "cudf::experimental::unary_op::TANH"
        ARCSINH "cudf::experimental::unary_op::ARCSINH"
        ARCCOSH "cudf::experimental::unary_op::ARCCOSH"
        ARCTANH "cudf::experimental::unary_op::ARCTANH"
        EXP "cudf::experimental::unary_op::EXP"
        LOG "cudf::experimental::unary_op::LOG"
        SQRT "cudf::experimental::unary_op::SQRT"
        CBRT "cudf::experimental::unary_op::CBRT"
        CEIL "cudf::experimental::unary_op::CEIL"
        FLOOR "cudf::experimental::unary_op::FLOOR"
        ABS "cudf::experimental::unary_op::ABS"
        RINT "cudf::experimental::unary_op::RINT"
        BIT_INVERT "cudf::experimental::unary_op::BIT_INVERT"
        NOT "cudf::experimental::unary_op::NOT"

    cdef extern unique_ptr[column] unary_operation(
        column_view input,
        unary_op op) except +

    cdef extern unique_ptr[column] is_null(column_view input) except +
    cdef extern unique_ptr[column] is_valid(column_view input) except +
    cdef extern unique_ptr[column] cast(
        column_view input,
        data_type out_type) except +
    cdef extern unique_ptr[column] is_nan(column_view input) except +
    cdef extern unique_ptr[column] is_not_nan(column_view input) except +
