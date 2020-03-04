# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.vector cimport vector

from cudf._libxx.types import np_to_cudf_types, cudf_to_np_types

from cudf._libxx.cpp.column.column cimport column
from cudf._libxx.cpp.table.table_view cimport table_view
cimport cudf._libxx.cpp.types as libcudf_types


cdef extern from "cudf/sorting.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[column] sorted_order(
        const table_view& table,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence) except +

    cdef bool is_sorted(
        table_view table,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence) except +
