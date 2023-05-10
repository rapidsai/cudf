# Copyright (c) 2020, NVIDIA CORPORATION.

from libc.stdint cimport int32_t
from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.types cimport data_type

ctypedef int32_t underlying_type_t_unary_op


cdef extern from "cudf/unary.hpp" namespace "cudf" nogil:

    ctypedef enum unary_operator:
        SIN "cudf::unary_operator::SIN"
        COS "cudf::unary_operator::COS"
        TAN "cudf::unary_operator::TAN"
        ARCSIN "cudf::unary_operator::ARCSIN"
        ARCCOS "cudf::unary_operator::ARCCOS"
        ARCTAN "cudf::unary_operator::ARCTAN"
        SINH "cudf::unary_operator::SINH"
        COSH "cudf::unary_operator::COSH"
        TANH "cudf::unary_operator::TANH"
        ARCSINH "cudf::unary_operator::ARCSINH"
        ARCCOSH "cudf::unary_operator::ARCCOSH"
        ARCTANH "cudf::unary_operator::ARCTANH"
        EXP "cudf::unary_operator::EXP"
        LOG "cudf::unary_operator::LOG"
        SQRT "cudf::unary_operator::SQRT"
        CBRT "cudf::unary_operator::CBRT"
        CEIL "cudf::unary_operator::CEIL"
        FLOOR "cudf::unary_operator::FLOOR"
        ABS "cudf::unary_operator::ABS"
        RINT "cudf::unary_operator::RINT"
        BIT_INVERT "cudf::unary_operator::BIT_INVERT"
        NOT "cudf::unary_operator::NOT"

    cdef extern unique_ptr[column] unary_operation(
        column_view input,
        unary_operator op) except +

    cdef extern unique_ptr[column] is_null(column_view input) except +
    cdef extern unique_ptr[column] is_valid(column_view input) except +
    cdef extern unique_ptr[column] cast(
        column_view input,
        data_type out_type) except +
    cdef extern unique_ptr[column] is_nan(column_view input) except +
    cdef extern unique_ptr[column] is_not_nan(column_view input) except +
