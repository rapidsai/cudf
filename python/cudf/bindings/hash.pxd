# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *


cdef extern from "cudf.h" nogil:

    ctypedef enum gdf_hash_func:
        GDF_HASH_MURMUR3=0,
        GDF_HASH_IDENTITY,

    cdef gdf_error gdf_hash_partition(
        int num_input_cols,
        gdf_column *input[],
        int columns_to_hash[],
        int num_cols_to_hash,
        int num_partitions,
        gdf_column *partitioned_output[],
        int partition_offsets[],
        gdf_hash_func hash
    ) except +

    cdef gdf_error gdf_hash(
        int num_cols,
        gdf_column **input,
        gdf_hash_func hash,
        uint32_t *initial_hash_values,
        gdf_column *output
    ) except +

    cdef gdf_error gdf_hash_columns(
        gdf_column **columns_to_hash,
        int num_columns,
        gdf_column *output_column,
        void *stream
    ) except +

