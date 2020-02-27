# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._libxx.lib cimport *


cdef extern from "cudf/quantiles.hpp" namespace "cudf::experimental" nogil:
    cdef unique_ptr[table] quantiles (
        table_view source_table,
        vector[double] q,
        interpolation interp,
        sorted is_input_sorted,
        vector[order] column_order,
        vector[null_order] null_precedence
    ) except +
