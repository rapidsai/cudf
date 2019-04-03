# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *

cdef extern from "reduction.hpp" namespace "cudf" nogil:

    cdef gdf_scalar reduction(gdf_column *inp, gdf_reduction_op op, gdf_dtype output_dtype) except +
    cdef void scan(gdf_column *inp, gdf_column *out, gdf_scan_op op, bool inclusive) except +
