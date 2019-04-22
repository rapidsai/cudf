# Copyright (c) 2019, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cudf.bindings.cudf_cpp cimport *


cdef extern from "cudf.h" nogil:

    # See cpp/include/cudf/io_types.h:22
    ctypedef enum gdf_input_type:
        FILE_PATH = 0,
        HOST_BUFFER,

    # See cpp/include/cudf/io_types.h:145
    ctypedef struct pq_read_arg:
        # Output Arguments - Allocated in reader
        int             num_cols_out
        int             num_rows_out
        gdf_column      **data
        int             *index_col

        # Input arguments
        gdf_input_type  source_type
        const char      *source
        size_t          buffer_size

        const char      **use_cols
        int             use_cols_len


    cdef gdf_error read_parquet(pq_read_arg *args) except +
