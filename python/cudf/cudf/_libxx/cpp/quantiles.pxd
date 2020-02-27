# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.vector cimport vector
from libcpp.memory cimport unique_ptr

from cudf._libxx.cpp.table.table cimport table
from cudf._libxx.cpp.table.table_view cimport table_view

cimport cudf._libxx.cpp.types as libcudf_types


cdef extern from "cudf/quantiles.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[table] quantiles (
        table_view source_table,
        vector[double] q,
        libcudf_types.interpolation interp,
        libcudf_types.sorted is_input_sorted,
        vector[libcudf_types.order] column_order,
        vector[libcudf_types.null_order] null_precedence
    ) except +
