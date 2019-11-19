# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *

from libcpp.utility cimport pair


cdef extern from "cudf/legacy/quantiles.hpp" namespace "cudf" nogil:

    ctypedef enum interpolation:
        LINEAR =0,
        LOWER,
        HIGHER,
        MIDPOINT,
        NEAREST,

    cdef gdf_error quantile_exact(
        gdf_column* col_in,
        interpolation prec,
        double q,
        gdf_scalar* result,
        gdf_context* ctxt
    ) except +

    cdef gdf_error quantile_approx(
        gdf_column* col_in,
        double q,
        gdf_scalar* result,
        gdf_context* ctxt
    ) except +

    cdef pair[cudf_table, cudf_table] group_quantiles(
        const cudf_table& key_table,
        const cudf_table& val_table,
        const vector[double]& quantiles,
        interpolation interpolation,
        bool ignore_nulls
    ) except +
