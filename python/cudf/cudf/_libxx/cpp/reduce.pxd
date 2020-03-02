from cudf._libxx.scalar cimport Scalar
from cudf._libxx.lib cimport column_view, data_type, scalar
from cudf._libxx.aggregation cimport aggregation
from libcpp.memory cimport unique_ptr
from cudf._libxx.cpp.column.column cimport (
    column
)

cdef extern from "cudf/reduction.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[scalar] cpp_reduce "cudf::experimental::reduce" (
        column_view col,
        const unique_ptr[aggregation] agg,
        data_type type
    )
    ctypedef enum scan_type:
        INCLUSIVE "cudf::experimental::scan_type::INCLUSIVE",
        EXCLUSIVE "cudf::experimental::scan_type::EXCLUSIVE",

    cdef unique_ptr[column] cpp_scan "cudf::experimental::scan" (
        column_view col,
        const unique_ptr[aggregation] agg,
        scan_type inclusive
    )
