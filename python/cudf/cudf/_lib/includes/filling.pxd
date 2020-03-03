# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf._lib.cudf cimport *


cdef extern from "cudf/legacy/filling.hpp" namespace "cudf" nogil:

    cdef cudf_table tile(
        const cudf_table & input,
        size_type count
    ) except +
