# Copyright (c) 2020, NVIDIA CORPORATION.

from cudf._libxx.lib import *
from cudf._libxx.lib cimport *

cdef extern from "cudf/sorting.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[column] sorted_order(
        table_view source_table,
        vector[order] column_order,
        vector[null_order] null_precedence) except +

cdef extern from "cudf/search.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[column] lower_bound(
        table_view source_table,
        table_view bins,
        vector[order] column_order,
        vector[null_order] null_precedence) except +

    cdef unique_ptr[column] upper_bound(
        table_view source_table,
        table_view bins,
        vector[order] column_order,
        vector[null_order] null_precedence) except +
