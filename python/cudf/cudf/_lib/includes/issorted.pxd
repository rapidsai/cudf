# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf._lib.cudf cimport *

cdef extern from "cudf/legacy/predicates.hpp" namespace "cudf" nogil:

    cdef bool is_sorted(
        cudf_table table,
        const vector[int8_t] descending,
        bool nulls_are_smallest
    ) except +
