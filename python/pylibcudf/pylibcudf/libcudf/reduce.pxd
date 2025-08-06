# Copyright (c) 2020-2025, NVIDIA CORPORATION.
from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport pair
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.aggregation cimport reduce_aggregation, scan_aggregation
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.types cimport data_type, null_policy
from rmm.librmm.cuda_stream_view cimport cuda_stream_view


cdef extern from "cudf/reduction.hpp" namespace "cudf" nogil:
    cdef unique_ptr[scalar] cpp_reduce "cudf::reduce" (
        column_view col,
        const reduce_aggregation& agg,
        data_type type,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cpdef enum class scan_type(bool):
        INCLUSIVE "cudf::scan_type::INCLUSIVE",
        EXCLUSIVE "cudf::scan_type::EXCLUSIVE",

    cdef unique_ptr[column] cpp_scan "cudf::scan" (
        column_view col,
        const scan_aggregation& agg,
        scan_type inclusive,
        null_policy null_handling,
        cuda_stream_view stream
    ) except +libcudf_exception_handler

    cdef pair[unique_ptr[scalar],
              unique_ptr[scalar]] cpp_minmax "cudf::minmax" (
        column_view col,
        cuda_stream_view stream
    ) except +libcudf_exception_handler
