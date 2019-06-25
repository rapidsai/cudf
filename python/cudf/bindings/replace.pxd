# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *

cdef extern from "replace.hpp" namespace "cudf" nogil:

    cdef gdf_column replace_nulls(const gdf_column& inp,
                                  const gdf_column& replacement_values) except +

    cdef gdf_column replace_nulls(const gdf_column& inp,
                                  const gdf_scalar& replacement_value) except +

    cdef gdf_column find_and_replace_all(const gdf_column &input_col,
                                         const gdf_column &values_to_replace,
                                         const gdf_column &replacement_values) except +