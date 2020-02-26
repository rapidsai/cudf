# Copyright (c) 2020, NVIDIA CORPORATION.
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._libxx.types import np_to_cudf_types, cudf_to_np_types

from cudf._libxx.includes.column.column cimport column
from cudf._libxx.includes.table.table_view cimport table_view
cimport cudf._libxx.includes.types as cudf_types


cdef extern from "cudf/sorting.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[column] sorted_order(
        table_view source_table,
        vector[cudf_types.order] column_order,
        vector[cudf_types.null_order] null_precedence) except +

cdef extern from "cudf/search.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[column] lower_bound(
        table_view source_table,
        table_view bins,
        vector[cudf_types.order] column_order,
        vector[cudf_types.null_order] null_precedence) except +

    cdef unique_ptr[column] upper_bound(
        table_view source_table,
        table_view bins,
        vector[cudf_types.order] column_order,
        vector[cudf_types.null_order] null_precedence) except +
