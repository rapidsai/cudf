from cudf._libxx.scalar cimport Scalar
from cudf._libxx.lib cimport *
from cudf._libxx.aggregation cimport *
from libcpp.memory cimport unique_ptr

cdef extern from "cudf/reduction.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[scalar] cpp_reduce "cudf::experimental::reduce" (
        column_view col,
        const unique_ptr[aggregation] agg,
        data_type type
    )
    cdef unique_ptr[scalar] cpp_scan "cudf::experimental::scan" (
        column_view col,
        const unique_ptr[aggregation] agg,
        bool inclusive
    )