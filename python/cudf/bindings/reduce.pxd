# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *


cdef extern from "reduction.hpp" namespace "cudf" nogil:
    cdef enum ReductionOp:
        Sum "cudf::ReductionOp::Sum"
        Min "cudf::ReductionOp::Min"
        Max "cudf::ReductionOp::Max"
        Product "cudf::ReductionOp::Product"
        SumOfSquares "cudf::ReductionOp::SumOfSquares"


    cdef enum ScanOp:
        Sum "cudf::ScanOp::Sum"
        Min "cudf::ScanOp::Min"
        Max "cudf::ScanOp::Max"
        Product "cudf::ScanOp::Product"
        SumOfSquares "cudf::ReductionOp::SumOfSquares"

    cdef gdf_scalar reduction(gdf_column *inp, ReductionOp op, gdf_dtype output_dtype) except +
    cdef void scan(gdf_column *inp, gdf_column *out, ScanOp op, bool inclusive) except +
