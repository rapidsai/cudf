# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.table.table_view cimport table_view

from cudf._lib.cpp.types cimport (
    interpolation,
    null_order,
    order_info,
    order,
    sorted,
)


cdef extern from "cudf/quantiles.hpp" namespace "cudf" nogil:

    cdef unique_ptr[column] quantile (
        column_view input,
        vector[double] q,
        interpolation interp,
        column_view ordered_indices,
        bool exact,
    ) except +

    cdef unique_ptr[table] quantiles (
        table_view source_table,
        vector[double] q,
        interpolation interp,
        sorted is_input_sorted,
        vector[order] column_order,
        vector[null_order] null_precedence,
    ) except +
