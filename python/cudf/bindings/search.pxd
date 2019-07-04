# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf.bindings.cudf_cpp cimport *

cdef extern from "search.hpp" namespace "cudf" nogil:

    cdef gdf_column lower_bound(const gdf_column& column,
        const gdf_column& values, bool nulls_as_largest) except +

    cdef gdf_column upper_bound(const gdf_column& column,
        const gdf_column& values, bool nulls_as_largest) except +

    cdef gdf_column lower_bound(const cudf_table& t, const cudf_table& values,
        bool nulls_as_largest) except +

    cdef gdf_column upper_bound(const cudf_table& t, const cudf_table& values,
        bool nulls_as_largest) except +
