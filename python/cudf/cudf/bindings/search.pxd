# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

# Copyright (c) 2018, NVIDIA CORPORATION.

from cudf.bindings.cudf_cpp cimport *

cdef extern from "cudf/search.hpp" namespace "cudf" nogil:

    cdef gdf_column lower_bound(
        const cudf_table& t,
        const cudf_table& values,
        vector[bool] c_desc_flags,
    ) except +

    cdef gdf_column upper_bound(
        const cudf_table& t,
        const cudf_table& values,
        vector[bool] c_desc_flags,
    ) except +
