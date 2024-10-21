# Copyright (c) 2020-2024, NVIDIA CORPORATION.
from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport data_type


cdef extern from "cudf/unary.hpp" namespace "cudf" nogil:

    cpdef enum class unary_operator(int32_t):
        SIN
        COS
        TAN
        ARCSIN
        ARCCOS
        ARCTAN
        SINH
        COSH
        TANH
        ARCSINH
        ARCCOSH
        ARCTANH
        EXP
        LOG
        SQRT
        CBRT
        CEIL
        FLOOR
        ABS
        RINT
        BIT_INVERT
        NOT

    cdef extern unique_ptr[column] unary_operation(
        column_view input,
        unary_operator op) except +libcudf_exception_handler

    cdef extern unique_ptr[column] is_null(
        column_view input
    ) except +libcudf_exception_handler
    cdef extern unique_ptr[column] is_valid(
        column_view input
    ) except +libcudf_exception_handler
    cdef extern unique_ptr[column] cast(
        column_view input,
        data_type out_type) except +libcudf_exception_handler
    cdef extern bool is_supported_cast(data_type from_, data_type to) noexcept
    cdef extern unique_ptr[column] is_nan(
        column_view input
    ) except +libcudf_exception_handler
    cdef extern unique_ptr[column] is_not_nan(
        column_view input
    ) except +libcudf_exception_handler
