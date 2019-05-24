# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *
from cudf.bindings.types cimport table as cudf_table

cdef extern from "stream_compaction.hpp" namespace "cudf" nogil:

    cdef gdf_column apply_boolean_mask(const gdf_column &input,
                                       const gdf_column &boolean_mask) except +

    cdef gdf_column drop_nulls(const gdf_column &input) except +

    cdef uint32_t gdf_get_unique_ordered_indices(const cudf_table& key_columns,
                                        gdf_index_type* unique_indices,
                                        const bool keep_first) except +

#cdef extern from "groupby.hpp" namespace "cudf" nogil:
#    cdef rmm::device_vector<gdf_index_type> gdf_unique_indices(cudf::table const& input_table,
#            gdf_context const& context) except +
