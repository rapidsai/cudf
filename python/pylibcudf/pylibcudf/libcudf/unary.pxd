# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libc.stdint cimport int32_t
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.types cimport data_type
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource


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
        BIT_COUNT
        BIT_INVERT
        NOT
        NEGATE

    cdef extern unique_ptr[column] unary_operation(
        column_view input,
        unary_operator op,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler

    cdef extern unique_ptr[column] is_null(
        column_view input,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
    cdef extern unique_ptr[column] is_valid(
        column_view input,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
    cdef extern unique_ptr[column] cast(
        column_view input,
        data_type out_type,
        cuda_stream_view stream,
        device_memory_resource* mr) except +libcudf_exception_handler
    cdef extern bool is_supported_cast(data_type from_, data_type to) noexcept
    cdef extern unique_ptr[column] is_nan(
        column_view input,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
    cdef extern unique_ptr[column] is_not_nan(
        column_view input,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler
