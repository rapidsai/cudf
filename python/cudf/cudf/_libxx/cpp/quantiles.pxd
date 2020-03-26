# Copyright (c) 2020, NVIDIA CORPORATION.

from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr

from cudf._libxx.cpp.column.column_view cimport column_view
from cudf._libxx.cpp.scalar.scalar cimport scalar
from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport table_view

from cudf._libxx.cpp.types cimport (
    interpolation,
    null_order,
    order_info,
    order,
    sorted,
)


cdef extern from "cudf/quantiles.hpp" namespace "cudf::experimental" nogil:

    cdef unique_ptr[scalar] quantile (
        column_view input,
        double q,
        interpolation interp,
        order_info column_order
    ) except +

    cdef unique_ptr[table] quantiles (
        table_view source_table,
        vector[double] q,
        interpolation interp,
        sorted is_input_sorted,
        vector[order] column_order,
        vector[null_order] null_precedence
    ) except +
