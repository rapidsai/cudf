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


cdef extern from "cudf/unary.hpp" namespace "cudf" nogil:

    ctypedef enum unary_op:
        SIN "cudf::unary_op::SIN"
        COS "cudf::unary_op::COS"
        TAN "cudf::unary_op::TAN"
        ARCSIN "cudf::unary_op::ARCSIN"
        ARCCOS "cudf::unary_op::ARCCOS"
        ARCTAN "cudf::unary_op::ARCTAN"
        SINH "cudf::unary_op::SINH"
        COSH "cudf::unary_op::COSH"
        TANH "cudf::unary_op::TANH"
        ARCSINH "cudf::unary_op::ARCSINH"
        ARCCOSH "cudf::unary_op::ARCCOSH"
        ARCTANH "cudf::unary_op::ARCTANH"
        EXP "cudf::unary_op::EXP"
        LOG "cudf::unary_op::LOG"
        SQRT "cudf::unary_op::SQRT"
        CBRT "cudf::unary_op::CBRT"
        CEIL "cudf::unary_op::CEIL"
        FLOOR "cudf::unary_op::FLOOR"
        ABS "cudf::unary_op::ABS"
        RINT "cudf::unary_op::RINT"
        BIT_INVERT "cudf::unary_op::BIT_INVERT"
        NOT "cudf::unary_op::NOT"

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
