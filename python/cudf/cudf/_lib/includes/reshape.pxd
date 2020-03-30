# Copyright (c) 2019-2020, NVIDIA CORPORATION.

from cudf._lib.cudf cimport *


cdef extern from "cudf/legacy/reshape.hpp" namespace "cudf" nogil:

    cdef gdf_column stack(
        const cudf_table & input
    ) except +
