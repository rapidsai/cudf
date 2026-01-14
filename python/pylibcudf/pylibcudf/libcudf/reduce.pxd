# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from libcpp cimport bool
from libcpp.functional cimport reference_wrapper
from libcpp.memory cimport unique_ptr
from libcpp.optional cimport optional
from libcpp.utility cimport pair
from pylibcudf.exception_handler cimport libcudf_exception_handler
from pylibcudf.libcudf.aggregation cimport reduce_aggregation, scan_aggregation, Kind
from pylibcudf.libcudf.column.column cimport column
from pylibcudf.libcudf.column.column_view cimport column_view
from pylibcudf.libcudf.scalar.scalar cimport scalar
from pylibcudf.libcudf.types cimport data_type, null_policy
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.librmm.memory_resource cimport device_memory_resource

ctypedef const scalar constscalar

cdef extern from "cudf/reduction.hpp" namespace "cudf" nogil:
    cdef unique_ptr[scalar] reduce(
        const column_view& col,
        const reduce_aggregation& agg,
        data_type output_type,
        optional[reference_wrapper[constscalar]] init,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cpdef enum class scan_type(bool):
        INCLUSIVE
        EXCLUSIVE

    cdef unique_ptr[column] scan(
        const column_view& col,
        const scan_aggregation& agg,
        scan_type inclusive,
        null_policy null_handling,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler

    cdef pair[unique_ptr[scalar], unique_ptr[scalar]] minmax(
        const column_view& col,
        cuda_stream_view stream,
        device_memory_resource* mr
    ) except +libcudf_exception_handler


cdef extern from "cudf/reduction.hpp" namespace "cudf::reduction" nogil:
    bool is_valid_aggregation(
        data_type source, Kind kind
    ) noexcept
