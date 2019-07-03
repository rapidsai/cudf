# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *


cdef extern from "reduction.hpp" nogil:

    ctypedef enum gdf_reduction_op:
      GDF_REDUCTION_SUM = 0,
      GDF_REDUCTION_MIN,
      GDF_REDUCTION_MAX,
      GDF_REDUCTION_PRODUCT,
      GDF_REDUCTION_SUMOFSQUARES,

    ctypedef enum gdf_scan_op:
      GDF_SCAN_SUM = 0,
      GDF_SCAN_MIN,
      GDF_SCAN_MAX,
      GDF_SCAN_PRODUCT,

cdef extern from "reduction.hpp" namespace "cudf" nogil:

    cdef gdf_scalar reduction(gdf_column *inp, gdf_reduction_op op, gdf_dtype output_dtype) except +
    cdef void scan(gdf_column *inp, gdf_column *out, gdf_scan_op op, bool inclusive) except +
