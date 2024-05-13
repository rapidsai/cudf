# Copyright (c) 2020-2024, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport pair

from cudf._lib.pylibcudf.libcudf.aggregation cimport (
    reduce_aggregation,
    scan_aggregation,
)
from cudf._lib.pylibcudf.libcudf.column.column cimport column
from cudf._lib.pylibcudf.libcudf.column.column_view cimport column_view
from cudf._lib.pylibcudf.libcudf.scalar.scalar cimport scalar
from cudf._lib.pylibcudf.libcudf.types cimport data_type


cdef extern from "cudf/reduction.hpp" namespace "cudf" nogil:
    cdef unique_ptr[scalar] cpp_reduce "cudf::reduce" (
        column_view col,
        const reduce_aggregation& agg,
        data_type type
    ) except +

    cpdef enum class scan_type(bool):
        INCLUSIVE "cudf::scan_type::INCLUSIVE",
        EXCLUSIVE "cudf::scan_type::EXCLUSIVE",

    cdef unique_ptr[column] cpp_scan "cudf::scan" (
        column_view col,
        const scan_aggregation& agg,
        scan_type inclusive
    ) except +

    cdef pair[unique_ptr[scalar],
              unique_ptr[scalar]] cpp_minmax "cudf::minmax" (
        column_view col
    ) except +
