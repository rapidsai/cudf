from cudf._libxx.lib cimport *

cdef extern from "cudf/search.hpp" namespace "cudf::experimental" nogil:

    cdef unique_ptr[column] lower_bound(
        table_view t,
        table_view values,
        vector[order] column_order,
        vector[null_order] null_precedence,
    )

    cdef unique_ptr[column] upper_bound(
        table_view t,
        table_view values,
        vector[order] column_order,
        vector[null_order] null_precedence,
    )

    cdef unique_ptr[column] contains(
        column_view haystack,
        column_view needles,
    )
