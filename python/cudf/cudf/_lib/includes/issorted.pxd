# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *

cdef extern from "cudf/legacy/predicates.hpp" namespace "cudf" nogil:

    cdef bool is_sorted(
        cudf_table table,
        const vector[int8_t] descending,
        bool nulls_are_smallest
    ) except +
