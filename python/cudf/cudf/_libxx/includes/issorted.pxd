# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._libxx.lib cimport *

cdef extern from "cudf/sorting.hpp" namespace "cudf::experimental" \
     nogil:

    cdef bool is_sorted(
        table_view table,
        vector[order] column_order,
        vector[null_order] null_precedence
    ) except +
