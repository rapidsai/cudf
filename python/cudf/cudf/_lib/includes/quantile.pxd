# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf._lib.cudf cimport *

from libcpp.utility cimport pair


cdef extern from "cudf/quantiles.hpp" namespace "cudf" nogil:

    cdef pair[cudf_table, cudf_table] group_quantiles(
        const cudf_table& key_table,
        const cudf_table& val_table,
        const vector[double]& quantiles,
        gdf_quantile_method interpolation,
        bool ignore_nulls
    ) except +
