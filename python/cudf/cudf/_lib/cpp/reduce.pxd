# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._lib.cpp.types cimport data_type
from cudf._lib.cpp.scalar.scalar cimport scalar
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.column.column cimport column
from cudf._lib.scalar cimport Scalar
from cudf._lib.aggregation cimport aggregation
from libcpp.memory cimport unique_ptr


cdef extern from "cudf/reduction.hpp" namespace "cudf" nogil:
    cdef unique_ptr[scalar] cpp_reduce "cudf::reduce" (
        column_view col,
        const unique_ptr[aggregation] agg,
        data_type type
    ) except +

    ctypedef enum scan_type:
        INCLUSIVE "cudf::scan_type::INCLUSIVE",
        EXCLUSIVE "cudf::scan_type::EXCLUSIVE",

    cdef unique_ptr[column] cpp_scan "cudf::scan" (
        column_view col,
        const unique_ptr[aggregation] agg,
        scan_type inclusive
    ) except +
